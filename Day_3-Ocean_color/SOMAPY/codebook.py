# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020

import numpy as np
import scipy as sp

from sklearn.decomposition import PCA
from decorators import timeit


class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass

def generate_hex_lattice(n_rows, n_columns):   # modifiée pour rcorrespondre à matlab
    x_coord = []
    y_coord = []
    for i in range(n_columns):
        for j in range(n_rows):
            x_coord.append(i+(j%2)/2.0)
            y_coord.append(np.sqrt(3.0)/2*j)
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates

def generate_hex_latticeold(n_rows, n_columns):  # ancienne version : ne correspond pas a matlab
    x_coord = []
    y_coord = []
    for i in range(n_columns):
        for j in range(n_rows):
            x_coord.append(i*1.5)
            y_coord.append(np.sqrt(2/3)*(2*j+(1+i)%2))
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates

def generate_rect_lattice(n_rows, n_columns): # Ajout LB pour correspondre à matlab
    x_coord = []
    y_coord = []
    for i in range(n_columns):
        for j in range(n_rows):
            x_coord.append(i)
            y_coord.append(j)
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates


class Codebook(object):

    def __init__(self, mapsize, lattice='rect'):
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2)))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = mapsize[0]*mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

        if lattice == "hexa":
            n_rows, n_columns = mapsize
            coordinates = generate_hex_lattice(n_rows, n_columns)
            #self.lattice_distances = (sp.spatial.distance_matrix(coordinates, coordinates)     # idem matlab
            #                          .reshape(n_rows * n_columns, n_rows, n_columns))
            
            self.lattice_distances =np.transpose(sp.spatial.distance_matrix(coordinates, coordinates).reshape(n_rows * n_columns, n_columns,n_rows),axes=(0,2,1))   # ajout LB
        elif lattice == 'rect':
            n_rows, n_columns = mapsize
            coordinates = generate_rect_lattice(n_rows, n_columns)
            #self.lattice_distances = (sp.spatial.distance_matrix(coordinates, coordinates)   # idem matlab
            #                          .reshape(n_rows * n_columns, n_rows, n_columns))
            self.lattice_distances= sp.spatial.distance_matrix(coordinates, coordinates).reshape(n_columns,n_rows,n_rows * n_columns).T
        else:
            raise
        

    @timeit()
    def random_initialization(self, data):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True
        
    def custom_initialization(self, data):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        self.matrix = np.loadtxt('initml.mat')
        self.initialized = True

    @timeit()
    def pca_linear_initialization(self, data,mask):
        """
        We initialize the map, just by using the first two first eigen vals and
        eigenvectors
        Further, we create a linear combination of them in the new map by
        giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication
        // PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen
        // vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected
        eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components) # RandomizedPCA is deprecated.
        #pca = PCA(n_components=pca_components, svd_solver='randomized')
        pca = PCA(n_components=pca_components, svd_solver='full')

        if mask is not None:
            pca.fit(data*mask)          # modif LB
        else:
            pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T


        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]
        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True

    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        return self.lattice_distances[node_ind]**2

    def _rect_dist(self, node_ind):
       
        return self.lattice_distances[node_ind]**2
        
       
