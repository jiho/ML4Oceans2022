# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020

from math import sqrt
from warnings import warn
import scipy
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from view import MatplotView
from plot_tools import coordinates_rectangularRegulargrid
from plot_tools import coordinates_hexagonalRegulargrid
from plot_tools import plot_hex_map
            
class UMatrixView(MatplotView):
    # ajouter pour le debogage
    #
    def _set_labels(self, cents, labels=None, ax=None, onlyzeros=False, fontsize=10, hex=False):
        if ax==None:
            ax=plt.gca()
        if ( not isinstance(labels,(list,tuple,np.ndarray))
             and labels==None ): # -> neuron number (a priori)
            if isinstance(cents,(list,tuple)):
                labels = np.array(range(len(cents)))
            elif isinstance(cents,np.ndarray):
                labels = np.array(range(cents.shape[0]))
            else:
                raise TypeError('unexpected cents type : {}'.format(type(cents)))
            #print('labels (default) :',labels)
        
        for i, txt in enumerate(labels):
            # gestion des types de labels
            if isinstance(txt,float):
                txt="{:5.1e}".format(txt)
            elif not isinstance(txt,(str,int)):
                raise TypeError('Unexpected type : {}'.format(type(val)))
            # suppression de zeros ?
            if onlyzeros == True:
                if txt !=  str(0) or txt != str(0.0) or "{:5.1e}".format(0.0):
                    txt = ""
                    
            #c = cents[i] if hex else (cents[i, 1] + 0.5, cents[-(i + 1), 0] + 0.5)
            #c = (cents[i, 0], cents[i, 1])  # modif LB
            c = (cents[i][0], cents[i][1])  # modif LB
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)


    def build_u_matrix(self, som, distance=1, row_normalized=False):  
        UD2 = som.calculate_map_dist()
        Umatrix = np.zeros((som.codebook.nnodes, 1))
        codebook = som.codebook.matrix
        if row_normalized:
            vector = som._normalizer.normalize_by(codebook.T, codebook.T).T     # modif LB
        else:
            vector = codebook

        for i in range(som.codebook.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            codebook_i = np.squeeze(np.asarray(codebook_i))         # ajout LB
            neighborbor_ind = UD2[i][0:] <= distance**2             # modif <= <   **2
            neighborbor_codebooks = vector[neighborbor_ind]
            codebook_i =np.expand_dims(codebook_i,0)             # Ajout LB
            #Umatrix[i] = scipy.spatial.distance_matrix(
            #    codebook_i, neighborbor_codebooks).mean()
            if som.mask is None:
                Umatrix[i] = (scipy.spatial.distance_matrix(codebook_i, neighborbor_codebooks).sum()
                                            /(len(neighborbor_ind)-1))            # modif LB
            else:
                Umatrix[i] = (scipy.spatial.distance_matrix(codebook_i*som.mask, neighborbor_codebooks*som.mask).sum()
                                            /(len(neighborbor_ind)-1))            # modif LB

        #return Umatrix.reshape(som.codebook.mapsize)
        
        return Umatrix.reshape(som.codebook.mapsize[1],som.codebook.mapsize[0]).T

    def show(self, som, distance=1, row_normalized=False, show_data=False,
             contooor=False, blob=False, labels=False,cmap=cm.jet,annotate=True):          
        umat = self.build_u_matrix(som, distance=distance,
                                   row_normalized=row_normalized)   
        labels_umat = umat.copy() # attention matrice 2D
        
        #print('umat=',umat)
        msz = som.codebook.mapsize
        #proj = som.project_data(som.data_raw) #modif LB
        proj = som.project_data(som._data)  #modif LB pas la peine de repartir des données brutes
        coord = som.bmu_ind_to_xy(proj)
        self._fig, ax = plt.subplots(1, 1,figsize=(self.width, self.height))
        if som.codebook.lattice=="rect":
            #print('labels_umat :\n',labels_umat.reshape(msz[0], msz[1],order='F'))
            #print('labels_umat.flatten() :',labels_umat.reshape(msz[0]*msz[1],1,order='F').flatten())
            #self._fig, ax = plt.subplots(1, 1,figsize=(self.width, self.height))
            #umat = np.flipud(np.fliplr(umat.reshape(msz[1], msz[0],order='C')))
            #umat = umat.reshape(msz[0], msz[1],order='F')
            #pl = plt.imshow(umat, cmap=plt.cm.get_cmap(cmap), alpha=1)
            pl = plt.imshow(labels_umat.reshape(msz[0], msz[1],order='F'),
                            cmap=plt.cm.get_cmap(cmap),
                            alpha=1)
            # Major ticks
            ax.set_xticks(np.arange(0, msz[1], 1))
            ax.set_yticks(np.arange(0, msz[0], 1))
            #ax.set_xticklabels(np.arange(1, msz[1]+1, 1))
            plt.setp(ax.get_xticklabels(), visible=False)
            #ax.set_yticklabels(np.arange(1, msz[0]+1, 1))
            plt.setp(ax.get_yticklabels(), visible=False)
            # Minor ticks
            ax.set_xticks(np.arange(-.5, msz[1]+.5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, msz[0]+.5, 1), minor=True)
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='k', linestyle='-', linewidth=3)
            ax.grid(which='major', visible=False,color='w', linestyle='-', linewidth=10)
            # on vire les ergots des ticks
            ax.tick_params(axis='both', which='both', length=0)
            # deux cas de figure pour la colorbar
            plt.colorbar(pl,fraction=.2,aspect=30, orientation="horizontal")
            #plt.colorbar(pl,fraction=.2,aspect=30, orientation="vertical")
            plt.title(self.title,fontsize=self.text_size)
            if annotate:
                cents=coordinates_rectangularRegulargrid(som.codebook.mapsize[0],
                                                         som.codebook.mapsize[1],
                                                         c=.5)
                self._set_labels(cents,
                                 ax=ax,
                                 labels=labels_umat.reshape(msz[0]*msz[1],1,order='F').flatten()
                                 #labels=labels_umat.flatten()
                )
        elif som.codebook.lattice=="hexa":
            #print('labels_umat :\n',labels_umat)
            #print('labels_umat.flatten() :',labels_umat.flatten())
            #raise NotImplementedError('mp doit etre mis en forme.')
            ax.axis('off') # requis a cause de la figure tracée avant
            ax, cents = plot_hex_map(labels_umat.flatten()[:,None],
                                     cmap=cmap,
                                     fig=self._fig,
                                     titles=['U-matrix'],
                                     msize=msz)  # ajout LB
            #print('cents :',cents)
            #print('labels :',labels_umat.flatten())
            if annotate:
                #warn("""Attention faut-il ajouter la ligne suivante ?
                #cents=coordinates_rectangularRegulargrid(som.codebook.mapsize[0],
                #som.codebook.mapsize[1],
                #c=.5)
                #(Attention, cela ne sera pas si simple.)""")
                self._set_labels(cents,
                                 ax=ax,
                                 labels=labels_umat.flatten()[::-1])
            #ax.set_frame_on(False)
            #plt.axis('off')
            #plt.box(False)
        else:
            raise ErrorValue('Unexpected lattice : "{}".'.format(som.codebook.lattice))
        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            plt.contour(umat, np.linspace(mn, mx, 15), linewidths=0.7,
                    cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray',
                        marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
            plt.axis('off')

        if labels:
            if labels is True:
                labels = som.build_data_labels()
            for label, x, y in zip(labels, coord[:, 1], coord[:, 0]):
                plt.annotate(str(label), xy=(x, y),
                             horizontalalignment='center',
                             verticalalignment='center')

        ratio = float(msz[0])/(msz[0]+msz[1])
        #self._fig.set_size_inches((1-ratio)*15, ratio*15)
        #plt.tight_layout()
        #plt.subplots_adjust(hspace=.00, wspace=.000)
        sel_points = list()

        if blob:
            from skimage.color import rgb2gray
            from skimage.feature import blob_log

            image = 1 / umat
            rgb2gray(image)

            # 'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * sqrt(2)
            plt.imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)
            sel_points = list()

            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2,
                               fill=False)
                ax.add_patch(c)
                dist = scipy.spatial.distance_matrix(
                    coord[:, :2], np.array([row, col])[np.newaxis, :])
                sel_point = dist <= r
                plt.plot(coord[:, 1][sel_point[:, 0]],
                         coord[:, 0][sel_point[:, 0]], '.r')
                sel_points.append(sel_point[:, 0])

        #
        #plt.show()
        return sel_points, umat
