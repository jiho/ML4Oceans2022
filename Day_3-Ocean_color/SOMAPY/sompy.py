# -*- coding: utf-8 -*-

# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020


# sorting impots [https://www.python.org/dev/peps/pep-0008/#imports]

import itertools # SOMMap::find_bmu, 
import logging   # SOMFactory::build, SOMMap::train, SOMMap::rough_train,
                 # SOMMap::finetune_train, SOMMap::_batchtrain
import os        # SOMMap::_batchtrain
import pickle    # SOMFactory::load, SOMMap::save
import shutil    # <- pas utilise ??? 
import tempfile  # SOMMap::_batchtrain
import types     # SOMMap::find_bmu

import matplotlib.pyplot as plt # SOMData::plot_tsne, SOMMap::plotplanes, SOMMap::plot_tsne
import networkx as nx # SOMMap::plotplanes,  SOMMap::plot_tsne 
import numpy as np # _chunk_based_bmu_find, _chunk_based_bmu_find2,
                   # SOMData::data_labels
                   # SOMMap::predict_probability, SOMMap::node_activation
                   # SOMMap::calculate_topographic_error, SOMMap::calculate_quantization_error
                   # SOMMap::calculate_map_size
import pandas as pd  # SOMData::plot_tsne
import seaborn as sns  # SOMData::plot_tsne
import sklearn.cluster as clust  # SOMMap::_cluster_kmeans

from collections import Counter  # SOMMap::node_labels_from_data
from multiprocessing.dummy import Pool # SOMMap::find_bmu
from multiprocessing import cpu_count  # SOMMap::find_bmu
from random import randint      # SOMMap::plot_tsne
from time import time           # SOMMap::_batchtrain
from warnings import warn       # cluster

from joblib import load, dump # SOMMap::_batchtrain
from joblib import Parallel, delayed # <- pas utilise ??? 
from scipy.sparse import csr_matrix  # SOMMap::update_codebook_voronoi
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # SOMMap::_cluster_hac
from sklearn.utils import gen_even_slices # <- pas utilise ??? 
from sklearn import neighbors # SOMMap::project_data, SOMMap::predict_by, 
                              # SOMMap::predict, SOMMap::find_k_nodes, 
                              # SOMMap::predict_probability, 
                              # SOMMap::node_activation
from sklearn.manifold import TSNE # SOMData::plot_tsne, SOMMap::plot_tsne
from sklearn.metrics import pairwise_distances # SOMMap::find_bmu
from sklearn.neighbors import NearestNeighbors # SOMMap::find_k_nodes


from codebook import Codebook
#from .codebook import Codebook
from decorators import timeit
#from .decorators import timeit
from neighborhood import NeighborhoodFactory
#from .neighborhood import NeighborhoodFactory
from normalization import NormalizerFactory
#from .normalization import NormalizerFactory

#from hyperlearn.metrics.cosine import cosine_distances as cos
#from hyperlearn.metrics.euclidean import euclidean_distances as euc


# remarques
# - remplacer np.partition et np.argpartition par torch.topk 
# - on the use of pool :
#    Ideally, you should call the Pool() constructor exactly once - not over & over again.
#    There are substantial overheads when creating worker processes, and you pay those costs
#    every time you invoke Pool(). The processes created by a single Pool() call stay around!
#    When they finish the work you've given to them in one part of the program, they stick around,
#    waiting for more work to do.
#    As to Pool.close(), you should call that when - and only when - you're never going to submit
#    more work to the Pool instance. So Pool.close() is typically called when the parallelizable
#    part of your main program is finished. Then the worker processes will terminate when all work
#    already assigned has completed.
#    It's also excellent practice to call Pool.join() to wait for the worker processes to terminate.
#    Among other reasons, there's often no good way to report exceptions in parallelized code
#    (exceptions occur in a context only vaguely related to what your main program is doing),
#    and Pool.join() provides a synchronization point that can report some exceptions that
#    occurred in worker processes that you'd otherwise never see.
#

# exceptions specifiques
class ComponentNamesError(Exception):
    pass
class LabelsError(Exception):
    pass


# A priori, les trois fonctions ci-dessous ont été sorties de leur classe pour prendre en compte les interactions entre les classes joblib et pickle ???
# A priori, cela n'est plus vrai. Il faudra voir si les fonctions ne doivent pas etre reintegrees a leurs classes.

# # Since joblib.delayed uses Pickle, this method needs to be a top level
# # method in order to be pickled
# # Joblib is working on adding support for cloudpickle or dill which will allow
# # class methods to be pickled
# # when that that comes out we can move this to SOM class
# #def _chunk_based_bmu_find(input_matrix, codebook, y2, nth=1):

# fonction utilisee dans SOMMap::find_bmus
def _chunk_based_bmu_find(input_matrix, codebook, y2, metric='euclidean',nth=1,mask=None):
    """
    Finds the corresponding bmus to the input matrix.

    :param input_matrix: a matrix of input data, representing input vector as
                         rows, and vectors features/dimention as cols
                         when parallelizing the search, the input_matrix can be
                         a sub matrix from the bigger matrix
    :param codebook: matrix of weights to be used for the bmu search
    :param y2: somme des carrés des codebooks (LB), la somme des carrés des individus X**2 est faite plus haut
    :param x2 sommes des carrée individus
    """
    x2 = input_matrix[1]
    input_matrix= input_matrix[0]
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    bmu = np.empty((dlen, 2))

    # It seems that small batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel
    blen = min(100, dlen) #essayer 1, 10, ....
    #blen = dlen
    i0 = 0
    if mask is not None:                # ajout LB
        input_matrix = input_matrix *mask
        
    while i0+1 <= dlen:
        low = i0
        high = min(dlen, i0+blen)
        i0 = i0+blen
        ddata = input_matrix[low:high+1]
        d = np.dot(codebook, ddata.T)
        d *= -2      # -2 X*Y                                    
        d += y2.reshape(nnodes, 1)  # + Y**2
        d += x2[low:high+1]    # +X**2
        bmu[low:high+1, 0] = np.argpartition(d, nth, axis=0)[nth-1]
        bmu[low:high+1, 1] = np.partition(d, nth, axis=0)[nth-1]     # norme au carré    
        del ddata

    return bmu


# fonction utilisee dans SOMMap::find_bmus
def _chunk_based_bmu_find2(input_matrix, codebook, y2, func, nth=1, mask=None):
    """
    Finds the corresponding bmus to the input matrix.

    :param input_matrix: a matrix of input data, representing input vector as
                         rows, and vectors features/dimention as cols
                         when parallelizing the search, the input_matrix can be
                         a sub matrix from the bigger matrix
    :param codebook: matrix of weights to be used for the bmu search
    :param y2: somme des carrés des codebooks (LB), la somme des carrés des individus X**2 est faite plus haut
    :param x2 sommes des carrée individus
    """
    input_matrix= input_matrix[0]
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    b = np.empty((dlen, 2))
    if mask is not None:                
        input_matrix = input_matrix *mask
    # It seems that small batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel

    D=func(input_matrix, codebook)

    #b = np.empty((dlen, 2))
    b[:, 0] = np.argpartition(D, nth, axis=1)[:,nth-1]
    b[:, 1] = np.partition(D, nth, axis=1)[:,nth-1] 

    return b


# fonction utilisee dans SOMData::data_labels_from_map et SOMData::node_labels_from_data
def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list


class SOMData(object):
    def __init__(self, data, component_names=None, data_labels = None, normalization = None, name = "sompyData"):
        """
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param labels:
        :param comp_names:
        :param comp_norm: 
        """
        
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        
        self._component_names = self.build_component_names() if component_names is None else component_names
        self._dlabel = self.build_data_labels() if data_labels is None else data_labels
        self.name = name
        if normalization and type(normalization) == str :
            self.component_normes = []
            normalizer = []
            for i in range(len(data[0])):
                self.component_normes.append(normalization)
                normalizer.append(NormalizerFactory.build(normalization))  
        elif normalization and type(normalization) == list :
            self._component_normes = normalization
            normalizer = []
            for i in range(len(data[0])):
                normalizer.append(NormalizerFactory.build(normalization[i]))
        else : normalizer = None
        if normalizer :
            for i in range(len(normalizer)):
                data[:,i] = normalizer[i].normalize(data[:,i])
            self._data = data
            self.isNormalized = True
        else:
            self._data = data
            self.isNormalized = False
        self._normalizer = normalizer if normalizer else None
        
    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list
        of strings
        """
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._dlabel = label
        
    def build_data_labels(self):
        return ['dlabel-' + str(i) for i in range(0, self._dlen)]
    
    def data_labels_from_map(self, sMap):
        """
        
        Set labels of nodes from a SomMap object
        Parameters
        ----------
        sMap : SomMap object.
                

        Returns
        -------
        None.

        """
        labels = np.empty(self._dlen).astype(str)
        bmu = sMap.find_bmu(sMap._data, njb = 1)
        for i in range(len(sMap._nlabel)):
            ind = get_index_positions(bmu[0], i)
            if ind != []:
                for k in ind :
                    labels[k] = sMap._nlabel[i]
        self._dlabel = labels
            
    
    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        """
        

        set the names of variables
        ----------
        compnames : any form, This includes lists, lists of tuples, tuples, 
        tuples of tuples, tuples of lists and ndarrays
            DESCRIPTION.

        Raises
        ------
        ComponentNamesError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print(compnames)
        
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Component names should have the same '
                                      'size as the data dimension/features')

    def build_component_names(self):
        return ['Variable-' + str(i+1) for i in range(0, self._dim)]
    
    def plot_tsne(self, init="pca", perplexity=10, verbose=2,mask=None):
        """
        
        visualize data in a 2d map
        Parameters
        ----------
        init : {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), The default is "pca"
            Initialization of embedding. Possible options are ‘random’, ‘pca’,
            and a numpy array of shape (n_samples, n_components). 
            PCA initialization cannot be used with precomputed distances and 
            is usually more globally stable than random initialization.
                        
        perplexity : float, default=10
            The perplexity is related to the number of nearest neighbors 
            that is used in other manifold learning algorithms. Larger 
            datasets usually require a larger perplexity. Consider selecting 
            a value between 5 and 50. Different values can result in 
            significantly different results.
        verbose : int, default=2
            Verbosity level.

        Returns
        -------
        None.

        """
        T_SNE = TSNE(2, init = init, perplexity = perplexity, verbose = verbose)
        if mask is None:
            x2d = T_SNE.fit_transform(self._data)
        else:
            x2d = T_SNE.fit_transform(self._data[:,mask>0])
        df = pd.DataFrame(self._data)
        df['label'] = self._dlabel
        
        rndperm = np.random.permutation(df.shape[0])
        df_subset = df.loc[rndperm[:df.shape[0]],:].copy()
        
        df_subset['tsne-2d-one'] = x2d[:,0]
        df_subset['tsne-2d-two'] = x2d[:,1]

        plt.figure()
        
        sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        #palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=1)
        plt.title('T-SNE des données')
        plt.show()

class SOMFactory(object): 

    @staticmethod
    def load(file,data=None):
        """
        

        load a SomMap object from a dictionary 
        ----------
        file : 'str'
            path of the dictionary.
        data : array, default=None
            DESCRIPTION. The default is None.

        Returns
        -------
        sm : SomMap
            a SomMap object containing data loaded from dictionary file.

        """
        #import pickle
        dico=pickle.load(open(file,'rb'))
       
        normalizer = None
        neighborhood_calculator = NeighborhoodFactory.build(dico['neighborhood'])
        
        sm=SOMMap(data, neighborhood_calculator,normalizer,mapsize=dico['mapsize'],mask=dico['mask'],mapshape=dico['mapshape'],
              lattice=dico['lattice'],initialization=dico['initialization'],training=dico['training'],
              radius_train=dico['radius_train'],name=dico['name'],component_names=dico['comp_names'],
              components_to_plot=None)
        if dico['normalization']:
            #normalizer = NormalizerFactory.build(dico['normalization'])
            sm._normalizer=dico['normalization']
        
        #sm._normalizer.params= dico['norm_params']
        #sm._normalizer.normalized = True if dico['normalization'] is not None else False
        sm.codebook.matrix = dico['codebook']
        sm.codebook.initialized=dico['codebookinitialized']
        sm._dim = dico['dim']
        sm.isNormalized=dico['isnormalised']
        #sm._bmu = dico['bmu']
        return sm
    def make_sData(data):
        return SOMData(data)
    @staticmethod
    def build(sData,
              mapsize=None,
              mask=None,
              mapshape='planar',
              lattice='rect',
              normalization='var',
              initialization='pca',
              neighborhood='gaussian',
              training='batch',
              radius_train='linear',
              name='sompyMap',
              components_to_plot=None,
              LogFile='logfile.log'):
        """
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)

        :param normalization: normalizer object calculator. Options are:
            - var

        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)

        :param lattice: type of lattice. Options are:
            - rect
            - hexa

        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random

        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        
        if LogFile is not None:
            logging.basicConfig(filename=LogFile, filemode='w', level=logging.INFO,force=True)
        logging.info('start')
        
        #sData= SOMData(Data)
        
        if normalization and type(normalization) == str :
            normalizer = []
            for i in range(len(sData._data[0])):
                normalizer.append(NormalizerFactory.build(normalization))  
        elif normalization and type(normalization) == list :
            normalizer = []
            for i in range(len(sData._data[0])):
                normalizer.append(NormalizerFactory.build(normalization[i]))
        else :
            normalizer = sData._normalizer
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        return SOMMap(sData._data, neighborhood_calculator, normalizer, mapsize, mask,
                   mapshape, lattice, initialization, training,radius_train, name, 
                   sData.component_names,components_to_plot, sData.isNormalized, sData._dlabel)


class SOMMap(object):


    def __init__(self,
                 data,
                 neighborhood,
                 normalizer=None,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 initialization='pca',
                 training='batch',
                 radius_train='linear',
                 name='sompy',
                 component_names=None,
                 components_to_plot=None,
                 isNormalized = False, 
                 data_labels= None):
        """
        Self Organizing Map

        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.
        :param normalizer: normalizer object calculator.
        :param mapsize: tuple/list defining the dimensions of the som. If
            single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som.
        :param lattice: type of lattice.
        :param initialization: method to be used for initialization of the som.
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        self.isNormalized=isNormalized
        if data is not None:           # ajout LB
            if normalizer and isNormalized == False : 
                for i in range(len(normalizer)):
                    data[:,i] = normalizer[i].normalize(data[:,i])
                self._data = data
                self.isNormalized=True
            else : 
                self._data = data
            self._data = self._data.astype('double')        # ajout LB
            self._dim = data.shape[1]
            self._dlen = data.shape[0]
            

        else:
            self._data = None
            self._dim = None
            self._dlen = None
        
        self._normalizer = normalizer
        #self._dlabel = self.build_data_labels() if data_labels is None else data_labels
        self._dlabel = data_labels
        self._bmu = None

        self.name = name
        self.data_raw = data
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization
        self.mask = mask 
        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize
        self.mapsize=mapsize
        self.codebook = Codebook(mapsize, lattice)
        self.training = training
        self.radius_train = radius_train
        self._component_names = self.build_component_names() if component_names is None else [component_names]
        self._distance_matrix = self.calculate_map_dist()
        self.components_to_plot=components_to_plot
        
    def __str__(self):
        return f'mapsize={self.mapsize}\nname={self.name}\nNormaliser={self._normalizer.name}\nMap shape={self.mapshape}'
    def attach_data(self,data,comp_names):          # ajout LB
        self.data_raw = data
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self.component_names=comp_names
        self._data = self._normalizer.normalize(data) 
        self._data = self._data.astype('double')     
        self._bmu = self.find_bmu(data, njb=1)
        
        
    def save(self,file):
        
        """
        
        Save the somMap in a dictionary file containing parameter fo the Map.
        
        Parameters
        ----------
        file : file, str, or pathlib.Path,  
            File or filename to which the data is saved. 
                .

        Returns
        -------
        None.

        """
        dico={'name':self.name,
              'codebook': self.codebook.matrix,
              'lattice':self.codebook.lattice,
              'mapsize':self.codebook.mapsize,
              'normalization':self._normalizer,
              'norm_params':self._normalizer.params,
              'comp_names':self._component_names[0],
              'mask':self.mask,
              'neighborhood':self.neighborhood.name,
              'codebookinitialized':self.codebook.initialized,
              'initialization':self.initialization,
              'bmu':self._bmu,
              'mapshape':self.mapshape,
              'training': self.training,
              'radius_train': self.radius_train,
              'dim':self._dim,
              'isnormalized':self.isNormalized}
        
        #import pickle
        pickle.dump(dico,open(file,'wb'))
    
    
    #def plotepoch(self,comp0,comp1):
    def plotplanes(self):            # ajout LB
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.components_to_plot is None:
            return
        comps=self.components_to_plot
        
        n={1:(1,1),2:(1,2),3:(2,2),4:(2,2)}     # lignes, colonnes
        nplots=len(comps)
        if nplots > 4:
            raise ValueError('Le nombre de comp doit etre inferieur ou égal à 4')
        nl=n[nplots][0]
        nc=n[nplots][1]
        
        neighbours_nodes = self.calculate_neighbours_nodes()
        edges = []
        for i in range(self.codebook.nnodes):
            for j in range(len(neighbours_nodes[i])):
               edges.append((i,neighbours_nodes[i][j]))
        nodes = [i for i in range(self.codebook.nnodes)]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        plt.clf()
        for i in range(nplots):
            refs = [self.codebook.matrix[:,comps[i][0]]],[self.codebook.matrix[:,comps[i][1]]]
            pos = {}
            for k in range(self.codebook.nnodes):
                name = int(k)
                pos[name] = (refs[0][0][k],refs[1][0][k])
            plt.subplot(nl,nc,i+1)
            #plt.scatter(self._data[:,comps[i][0]],self._data[:,comps[i][1]],marker='x',c='b')
            #plt.scatter([self.codebook.matrix[:,comps[i][0]]],[self.codebook.matrix[:,comps[i][1]]],marker='o',c='r')
            nx.draw_networkx(G,pos,node_color ='r',arrows=False, with_labels=False, node_size = 50)
            plt.scatter(self._data[:,comps[i][0]],self._data[:,comps[i][1]],marker='x',c='b')
            plt.xlabel(self._component_names[0][comps[i][0]])
            plt.ylabel(self._component_names[0][comps[i][1]])
            plt.title('comp. {} - {}'.format(comps[i][0],comps[i][1]))
            
        plt.pause(0.1)

    def plot_tsne(self, init="pca", perplexity=10, verbose=2,n_iter=1200,mask=None):
        """
        
        visualize the map in a 2d plan
        
        Parameters
        ----------
        init : {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default='pca’
            Initialization of embedding. Possible options are ‘random’, ‘pca’,
            and a numpy array of shape (n_samples, n_components). 
            PCA initialization cannot be used with precomputed distances and 
            is usually more globally stable than random initialization.
            
        perplexity : float, default=10
            The perplexity is related to the number of nearest neighbors that
            is used in other manifold learning algorithms. Larger datasets 
            usually require a larger perplexity. Consider selecting a value 
            between 5 and 50. Different values can result in significantly 
            different results.

        verbose : int, default=2
            Verbosity level.


        Returns
        -------
        None.

        """
        self.node_labels_from_data(self._dlabel)
        
        T_SNE = TSNE(2, init = init, perplexity = perplexity, verbose = verbose,n_iter=n_iter)
        if mask is None and self.mask is not None:
            x2d = T_SNE.fit_transform(self.codebook.matrix[:,self.mask>0])
        elif mask is not None:
            x2d = T_SNE.fit_transform(self.codebook.matrix[:,mask>0])
        else:
            x2d = T_SNE.fit_transform(self.codebook.matrix)

        neighbours_nodes = self.calculate_neighbours_nodes()
        edges = []
        for i in range(self.codebook.nnodes):
            for j in range(len(neighbours_nodes[i])):
               edges.append((i,neighbours_nodes[i][j]))
        nodes = [i for i in range(self.codebook.nnodes)]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i in range(self.codebook.nnodes):
            G.nodes[i]['label'] = self._nlabel[i]
        G.add_edges_from(edges)
        
        pos ={}
        for i in range(self.codebook.nnodes):
            name = int(i)
            pos[name]=(x2d[i][0],x2d[i][1])
        
        colorlist = []
        for i in range(len(np.unique(self._nlabel))):
            colorlist.append('#%06X' % randint(0, 0xFFFFFF))
        
        plt.figure()
        for i in range(len(np.unique(self._nlabel))):
            nodelist = []
            for j in range(self.codebook.nnodes):
                if G.nodes[j]['label'] == np.unique(self._nlabel)[i]:
                    nodelist.append(j)
            nx.draw_networkx(G,pos,arrows=False,nodelist = nodelist, node_color = colorlist[i], 
                             with_labels=False, node_size = 50, label = np.unique(self._nlabel)[i])
        #plt.scatter(x2d[:, 0], x2d[:, 1],s=5,marker='o')
        plt.xlabel('tsne-2d-one')
        plt.ylabel('tsne-2d-two')
        plt.title('T-SNE des neurones référants')
        plt.show()
    
    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        """
        set names of variables 
        Parameters
        ----------
        compnames : list of string
            list containing names of components to be set.

        Raises
        ------
        ComponentNamesError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Component names should have the same '
                                      'size as the data dimension/features')

    def build_component_names(self):
        return ['Variable-' + str(i+1) for i in range(0, self._dim)]
    
    @property
    def node_labels(self):
        return self._nlabel

    @node_labels.setter
    def node_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list
        of strings
        """
        
        if labels.shape == (1, self.codebook.nnodes):
            label = labels.T
        elif labels.shape == (self.codebook.nnodes, 1):
            label = labels
        elif labels.shape == (self.codebook.nnodes,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._nlabel = label
    
    def build_data_labels(self):
        return ['dlabel-' + str(i) for i in range(0, self._dlen)]
    
    def build_node_labels(self):
        return ['nlabel-' + str(i) for i in range(0, self.codebook.nnodes)]
    
    def node_labels_from_data(self, labels):
        """
        set labels of the the nodes, from labels of data

        Parameters
        ----------
        labels : list
            labels of data.

        Returns
        -------
        None.

        """
        nlabels = [] 
        for i in range(self.codebook.nnodes):
            ind = get_index_positions(self._bmu[0], i)
            if ind != []:
                subData = [labels[k] for k in ind]
                nlabels.append(Counter(subData).most_common(1)[0][0])
            else:
                nlabels.append("Nan")
        self._nlabel = nlabels

    def calculate_map_dist(self):                                       # CALCUL MATRICE DIST
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            #distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
            distance_matrix[i] = self.codebook.grid_dist(i).T.reshape(1, nnodes)    #attention c'et la distance au carré
        return distance_matrix

    @timeit()
    def train(self,
              n_job=1,
              shared_memory=False,
              verbose='info',
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf,
              alreadyinit=False,
              watch_evolution = 1,
              metric='euclidean'):
        """
        Trains the som

        :param n_job: number of jobs to use to parallelize the traning
        :param shared_memory: flag to active shared memory
        :param verbose: verbosity, could be 'debug', 'info' or None
        :param train_len_factor: Factor that multiply default training lenghts
            (similar to "training" parameter in the matlab version). (lbugnon)
        """
        # logging.root.setLevel(
        #     getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info(" Training...")
        print('Training ...')
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            "      > number of jobs in parallel: {n_job}\n"
            " -------------------------------------------------------------\n")
            .format(data_len=self._dlen,
                    data_dim=self._dim,
                    mpsz0=self.codebook.mapsize[0],
                    mpsz1=self.codebook.mapsize[1],
                    array_size=np.log10(
                        self._dlen * self.codebook.nnodes * self._dim),
                    n_job=n_job))

        if self.initialization == 'random' and alreadyinit ==False:
            self.codebook.random_initialization(self._data)
            
        elif self.initialization == 'custom' and alreadyinit ==False:
            self.codebook.custom_initialization(self._data)

        elif self.initialization == 'pca'and alreadyinit ==False:
            self.codebook.pca_linear_initialization(self._data,self.mask)
        elif alreadyinit ==False:
             raise AttributeError ('initialisation inconnue')
        if train_rough_len >0:
            self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,
                         trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen, watch_evolution = watch_evolution,
                         metric=metric)
        
        if train_finetune_len >0:
            self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,
                            trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen, watch_evolution = watch_evolution, 
                            metric = metric)

        
        if self._bmu is None:       # calcul bmu meme si pas entrainé
            self._bmu = self.find_bmu(self._data, njb=n_job, metric=metric)
        #self.plotplanes2()
        logging.debug(
            " --------------------------------------------------------------")
        logging.info(" Final quantization error: %f" % np.mean(self._bmu[1]))
        
    
    #@timeit()
    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    @timeit()    
    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,
                    trainlen_factor=1,maxtrainlen=np.Inf, watch_evolution = 0, metric= 'euclidean'):

        logging.info(" Rough training...")
        print(" Rough training...")

        ms, mpd = self._calculate_ms_and_mpd()
        #lbugnon: add maxtrainlen
        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen
        #print("maxtrainlen %d",maxtrainlen)
        #lbugnon: add trainlen_factor
        trainlen=int(trainlen*trainlen_factor)
        
        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory, watch_evolution, metric= metric)
    
    @timeit()
    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, 
                       radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf, watch_evolution=0, metric= 'euclidean'):
        logging.info(" Finetune training...")
        print('Finetune training')
        ms, mpd = self._calculate_ms_and_mpd()

        #lbugnon: add maxtrainlen
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin # from radius fin in rough training
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin # max(1, ms/128)

        #print("maxtrainlen %d",maxtrainlen)
        
        #lbugnon: add trainlen_factor
        trainlen=int(trainlen_factor*trainlen)
        
            
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory, watch_evolution = watch_evolution, metric=metric)

    @timeit()
    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False, watch_evolution = 1, metric='euclidean'):
        
        if self.radius_train == 'linear':
            radius = np.linspace(radiusin, radiusfin, trainlen)
            
        elif self.radius_train == 'power_series':
            radius = []
            ratio = radiusfin / radiusin
            for i in range(trainlen):
                radius.append(radiusin * ((ratio)**(i/trainlen)))
                
        elif self.radius_train == 'inverse_of_time' :
            radius = []
            B = trainlen / ((radiusin / radiusfin) - 1)
            A = B * radiusin
            for i in range(trainlen):
                radius.append(A/(i+B))
                
        else :
            raise AttributeError ('évolution du radius inconnue')

        if shared_memory:
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None

        # X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use
        # for each data row in bmu finding.
        # Since it is a fixed value we can skip it during bmu finding for each
        # data point, but later we need it calculate quantification error
        

        logging.info(" radius_ini: %f , radius_final: %f, trainlen: %d\n" %
                     (radiusin, radiusfin, trainlen))
        print("radius_ini: {:.3f} , radius_final: {:.3f}, trainlen: {}\n".format(radiusin, radiusfin, trainlen))
        for i in range(trainlen):
            t1 = time()
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            bmu = self.find_bmu(data, njb=njob, metric=metric)
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu,
                                                                neighborhood)

            #print('nombre de neurones activés : ',len(np.unique(bmu[0])))
            qerror=self.calculate_quantization_error(njob=njob, metric=metric,mask=self.mask)
            terror = self.calculate_topographic_error(njob=njob, metric=metric)
            
            print('Epoch : {} qErr : {:.4f}  tErr : {:.4f} radius: {:.1f}'.format(i,qerror,terror,radius[i]))
            
            logging.info(
                " epoch: {} ---> elapsed time:  {:2.2f}, quantization error: {:2.4f}".format(i,time()-t1,qerror))
            if np.any(np.isnan(qerror)):
                logging.info("nan quantization error, exit train\n")
                
                #sys.exit("quantization error=nan, exit train")
            
            if watch_evolution >0 and i%watch_evolution ==0 :
                self.plotplanes()        # ajout LB
        if watch_evolution == 0 :
            self.plotplanes()  
        #print('bmu = {}'.format(bmu[1] + fixed_euclidean_x2))
        bmu = self.find_bmu(data, njb=njob, metric=metric)                     # ajout LB : il faut remettre à jour
        #tmp= bmu[1] + fixed_euclidean_x2
        #tmp[tmp<0]=0        # ajout LB
        #bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)
        #bmu[1]=tmp
        self._bmu = bmu

    #@timeit()
    def find_bmu(self, input_matrix, njb=1, nth=1, metric='euclidean'):
        """
        Finds the best matching unit (bmu) for each input data from the input
        matrix. It does all at once parallelizing the calculation instead of
        going through each input and running it against the codebook. you can 
        change the metric to measure distance        

        Parameters
        ----------
        input_matrix : ndarray of shape (n_samples, n_features)
            numpy matrix representing inputs as rows and features/dimension 
            as cols.
        njb : int, default=1
            number of jobs to parallelize the search.
        nth : int, default=1
            DESCRIPTION. The default is 1.
        metric : str or function, default='euclidean'
            metric to use for measuring the distance in finding bmu. valid 
            values for metric are: ['cityblock', 'cosine', 'euclidean', 'l1', 
            'l2','manhattan'], or any function that
            measures distance between two elements.

        Raises
        ------
        LabelsError
            DESCRIPTION.

        Returns
        -------
        ndarray of shape (2,n_samples)
            the best matching unit for each input containing the bmu for each 
            element and the distance between bmu and the element.

        """
        dlen = input_matrix.shape[0]
        if self.mask is not None:                # ajout LB
            codebookmask =  self.codebook.matrix*self.mask.squeeze()
            datamask = input_matrix*self.mask
        else:
            codebookmask = self.codebook.matrix
            datamask = input_matrix

        if njb == -1:
            njb = cpu_count()
        x2=np.empty((input_matrix.shape[0],))
        y2=np.empty((self.codebook.matrix.shape[0],))
            
        if isinstance(metric, types.FunctionType) or (metric == 'euclidean'):
            #pool = Pool(njb)       # Ne calcule-t-on pas les x2 et y2 trop de fois (ac)
            with Pool(njb) as pool: # As-tu rate quelque chose ? (ac) 
                if metric == 'euclidean':
                    y2 = np.einsum('ij,ij->i', self.codebook.matrix, codebookmask)  # somme des carrés Y**2
                    x2 = np.einsum('ij,ij->i', datamask, input_matrix) ##multiplication élément par élément, et sommation des lignes du produit
                    chunk_bmu_finder = _chunk_based_bmu_find 
                elif isinstance(metric, types.FunctionType) :
                    chunk_bmu_finder = (lambda input_matrix, codebook, y2, nth, mask
                                    : _chunk_based_bmu_find2(input_matrix, codebook, y2, metric, nth, mask))
                # fonctions utilitaires (mises ici parce qu'erreur quand sorties)
                # elles utilises des variables locales a la fonction appelante
                def __row_chunk(part): # pour SOMMap::find_bmu
                    return part * dlen // njb
                def __col_chunk(part): # pour SOMMap::find_bmu
                    return min((part+1)*dlen // njb, dlen)

                chunks = [[datamask[__row_chunk(i):__col_chunk(i)],x2[__row_chunk(i):__col_chunk(i)]] for i in range(njb)] 
                b = pool.map(lambda chk : chunk_bmu_finder(input_matrix=chk, codebook=codebookmask, y2=y2, nth=nth,mask=self.mask), chunks)  # modif LB
            #pool.close()
            #pool.join()
            bmu = np.asarray(list(itertools.chain(*b))).T          
            del b
            
        elif isinstance(metric, str) and metric != 'euclidean':
            D=pairwise_distances(datamask, codebookmask, metric=metric, n_jobs=njb)
            b = np.empty((dlen, 2))
            b[:, 0] = np.argpartition(D, nth, axis=1)[:,nth-1]
            b[:, 1] = np.partition(D, nth, axis=1)[:,nth-1] 
            bmu=b.T
            del b            
        else : 
            raise LabelsError('wrong metric format')
        
        return bmu

    #@timeit()
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        """
        Updates the weights of each node in the codebook that belongs to the
        bmu's neighborhood.

        First finds the Voronoi set of each node. It needs to calculate a
        smaller matrix.
        Super fast comparing to classic batch training algorithm, it is based
        on the implemented algorithm in som toolbox for Matlab by Helsinky
        University.

        :param training_data: input matrix with input vectors as rows and
            vector features as cols
        :param bmu: best matching unit for each input data. Has shape of
            (2, dlen) where first row has bmu indexes
        :param neighborhood: matrix representing the neighborhood of each bmu

        :returns: An updated codebook that incorporates the learnings from the
            input data
        """
        row = bmu[0].astype(int)              # pour chape individu le numero du codebook associé
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)           # autant de 1 que d'individus
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))     # nbr codebook x nbr individus avec des 1 lorsque un individu voisinage du codebook, 0 sinon
        S = P.dot(training_data)     # nbr codebook x dim codebook : formule 5 page 11 de somtoolbox 5 matlab

        # neighborhood has nnodes*nnodes and S has nnodes*dim
        # ---> Nominator has nnodes*dim
        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)  # role de la transposée ???
        new_codebook = np.divide(nom, denom)
        if (denom==0.0).sum() >0:
            print('denominateur nul',denom)
            raise
        #return np.around(new_codebook, decimals=6)
        return np.asarray(new_codebook)                            # modif LB
    
    #@timeit()
    def project_data(self, data,normalize=False): # modif LB
        """
        Projects a data set to a trained SOM. It is based on nearest
        neighborhood search module of scikitlearn, but it is not that fast.
        """
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        labels = np.arange(0, self.codebook.matrix.shape[0])
        clf.fit(self.codebook.matrix, labels)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        if normalize:
            data = self._normalizer.normalize_by(self.data_raw, data)

        return clf.predict(data)
    #@timeit()
    def predict_by(self, data, target, k=5, wt='distance'):
        # here it is assumed that target is the last column in the codebook
        # and data has dim-1 columns
        print('fonction predict_by est elle utilisée ?')
        raise
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != target]
        x = self.codebook.matrix[:, indX]
        y = self.codebook.matrix[:, target]
        n_neighbors = k
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indX]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indX], data)

        predicted_values = clf.predict(data)
        predicted_values = self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)
        return predicted_values
    
    #@timeit()
    def predict(self, x_test, k=5, wt='distance'):
        """
        Similar to SKlearn we assume that we have X_tr, Y_tr and X_test. Here
        it is assumed that target is the last column in the codebook and data
        has dim-1 columns

        :param x_test: input vector
        :param k: number of neighbors to use
        :param wt: method to use for the weights
            (more detail in KNeighborsRegressor docs)
        :returns: predicted values for the input data
        """
        
        target = self.data_raw.shape[1]-1
        x_train = self.codebook.matrix[:, :target]
        y_train = self.codebook.matrix[:, target]
        clf = neighbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        x_test = self._normalizer.normalize_by(
            self.data_raw[:, :target], x_test)
        predicted_values = clf.predict(x_test)

        return self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)

    #@timeit()
    def find_k_nodes(self, data, k=5):
        """
        
        Finds the k most similar nodes to the input vector
        Returns indices of and distances to the neighbors of each point.
        
        Parameters
        ----------
        data : ndarray
            input data.
        k : int, default=5
            Number of neighbors.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        
        """
        #from sklearn.neighbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        neighbor = NearestNeighbors(n_neighbors=k)
        neighbor.fit(self.codebook.matrix)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        return neighbor.kneighbors(
            self._normalizer.normalize_by(self.data_raw, data))

    #@timeit()
    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding
        matrix x,y coordinates.

        :param bmu_ind: node index of the best matching unit
            (number of node from top left node)
        :returns: corresponding (x,y) coordinate
        """
        rows = self.codebook.mapsize[0]
        cols = self.codebook.mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)
    
    #@timeit()
    def cluster(self, n_clusters=8):
        """
        Appel la fonction _cluster_kmeans 
        """
        warn('Careful it uses Kmeans and not HAC. It is sensitive to initialization.')
        return self._cluster_kmeans(n_clusters=n_clusters)

    def _cluster_hac(self,n_clusters=8,method='ward', metric='euclidean', optimal_ordering=False):
        Z = linkage(self.codebook.matrix[:,:],
                    method=method,
                    metric=metric,
                    optimal_ordering=optimal_ordering)
        cl_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        self.cluster_labels = cl_labels
        return cl_labels
    
    def _cluster_kmeans(self,n_clusters=8):
        """
        K-Means clustering of map nodes.

        Compute cluster centers and predict cluster index for each node.
        
        Parameters
        ----------
        n_clusters : int, default=8
            The number of clusters to form as well as the number of centroids
            to generate.

        
        Returns
        -------
        cl_labels : ndarray of shape (n_nodes,)
            Index of the cluster each node belongs to.

        """
        #import sklearn.cluster as clust
        #cl_labels = clust.KMeans(n_clusters=n_clusters).fit_predict(
        #    self._normalizer.denormalize_by(self.data_raw,
        #                                    self.codebook.matrix))
        cl_labels = clust.KMeans(n_clusters=n_clusters).fit_predict(self.codebook.matrix)
        
        #print(cl_labels)
        self.cluster_labels = cl_labels
        return cl_labels

    #@timeit()    
    def predict_probability(self, data, target, k=5):
        """
        Predicts probability of the input data to be target

        :param data: data to predict, it is assumed that 'target' is the last
            column in the codebook, so data hould have dim-1 columns
        :param target: target to predict probability
        :param k: k parameter on KNeighborsRegressor
        :returns: probability of data been target
        """
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indx = ind[ind != target]
        x = self.codebook.matrix[:, indx]
        y = self.codebook.matrix[:, target]

        clf = neighbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indx]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indx], data)

        weights, ind = clf.kneighbors(data, n_neighbors=k,
                                      return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self.codebook.matrix[ind, target])
        labels[labels >= 0] = 1

        # for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        # for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        return np.concatenate((pos_prob, neg_prob), axis=1)

    #@timeit()
    def node_activation(self, data, target=None, wt='distance'):
        """
        
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            input data.
        target : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        weights : TYPE
            DESCRIPTION.
        ind : TYPE
            indices of .

        """
        weights, ind = None, None

        if not target:
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.codebook.nnodes)
            labels = np.arange(0, self.codebook.matrix.shape[0])
            clf.fit(self.codebook.matrix, labels)

            # The codebook values are all normalized
            # we can normalize the input data based on mean and std of
            # original data
            data = self._normalizer.normalize_by(self.data_raw, data)
            weights, ind = clf.kneighbors(data)

            # Softmax function
            weights = 1./weights

        return weights, ind

    #@timeit()
    def calculate_topographic_error(self, njob=1, metric= 'euclidean'):
        bmus1 = self.find_bmu(self.data_raw, njb=njob, nth=1, metric= metric)
        bmus2 = self.find_bmu(self.data_raw, njb=njob, nth=2, metric= metric)
        topographic_error = None
        if self.codebook.lattice=="rect":
            bmus_gap = np.abs((self.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - self.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
            topographic_error = np.mean(bmus_gap != 1)
        elif self.codebook.lattice=="hexa":
            dist_matrix_1 = self.codebook.lattice_distances[bmus1[0].astype(int)].reshape(len(bmus1[0]), -1)
            topographic_error = (np.array(
                [distances[bmu2] for bmu2, distances in zip(bmus2[0].astype(int), dist_matrix_1)]) > 2).mean()
        return(topographic_error)

    #@timeit()
    # Ajouter masque e
    def calculate_quantization_error(self, njob=1, metric='euclidean',mask=None):
        neuron_values = self.codebook.matrix[self.find_bmu(self._data, njb=njob, metric=metric)[0].astype(int)]
        #quantization_error = np.mean(np.abs(neuron_values - self._data))   # norme L1 et pas L2 ????
        if mask is None and self.mask is not None:
            quantization_error = np.mean(((neuron_values - self._data)*self.mask)**2) *np.sum(self.mask)
        elif mask is not None:
            quantization_error = np.mean(((neuron_values - self._data)*mask)**2) *np.sum(mask)
        else:
            quantization_error = np.mean((neuron_values - self._data)**2) *neuron_values.shape[1]  # norme L1 et pas L2 ????
        return quantization_error

    #@timeit()
    def calculate_map_size(self, lattice):
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        D = self.data_raw.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        VS = np.linalg.eig(A)
        eigval = sorted(VS[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            size1 = min(munits, round(np.sqrt(munits / ratio)))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]


    #@timeit()
    def calculate_neighbours_nodes(self):
        res = []
        for i in range(self.codebook.nnodes):
            current = []
            for j in range(self.codebook.nnodes):
                if (self._distance_matrix[i][j] < 1.5 and self._distance_matrix[i][j] != 0):
                    current.append(j)
            res.append(current)
        return res


