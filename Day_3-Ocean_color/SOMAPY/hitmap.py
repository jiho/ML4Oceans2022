# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from view import MatplotView
from mapview import MapView

from plot_tools import plot_hex_map
from plot_tools import coordinates_rectangularRegulargrid
from plot_tools import coordinates_hexagonalRegulargrid

#from codebook import generate_rect_lattice, generate_hex_lattice



class HitMapView(MapView):

    def _set_labels(self, cents, ax=None, labels=None, onlyzeros=False, fontsize=7, hex=False):
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
        
        #print('cents :\n',cents)
        for i, txt in enumerate(labels):
            if onlyzeros == True:
                if txt > 0:
                    txt = ""
            #c = cents[i] if hex else (cents[i, 1], cents[-(i + 1), 0])
            #c = (cents[i, 0], cents[i, 1])  # modif LB
            c = (cents[i][0], cents[i][1])  # modif LB
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)
            #print(txt, c)

    def show(self, som, data=None, anotate=True, labels=None, onlyzeros=False, labelsize=7, cmap=cm.jet):
        if False:
            org_w = self.width
            org_h = self.height
            (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
             axis_num) = self._calculate_figure_params(som, 1, 1)
            self.width /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
            self.height /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        try:
            clusters = getattr(som, 'cluster_labels')
            #print('using existing labels')
        except:
            print('process kmean clustering')
            clusters = som.cluster()

        if labels is None and anotate==True:
            labels = [str(c) for c in clusters]
            #print('labels : ',labels)
            
        # codebook = getattr(som, 'cluster_labels', som.cluster())
        msz = som.codebook.mapsize

        #self.prepare()
        self._fig, ax = plt.subplots(1,1,figsize=(self.width, self.height))
        if som.codebook.lattice == "rect":
            #clusters = np.flipud(np.fliplr(clusters.reshape(msz[1], msz[0],order='C')))
            clusters = clusters.reshape(msz[0], msz[1],order='F')
            pl = plt.imshow(clusters, alpha=1)
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
            ax.grid(which='minor', visible=True, color='k', linestyle='-', linewidth=3)
            ax.grid(which='major', visible=False,color='w', linestyle='-', linewidth=10)
            # on vire les ergots des ticks
            ax.tick_params(axis='both', which='both', length=0)
            # deux cas de figure pour la colorbar
            plt.colorbar(pl,fraction=.2,aspect=30, orientation="horizontal")
            #plt.colorbar(pl,fraction=.2,aspect=30, orientation="vertical")
            #plt.colorbar(pl)#,aspect=100, orientation="horizontal")
            plt.title(self.title,fontsize=self.text_size)
            #if data:
            #    proj = som.project_data(data)
            #    cents = som.bmu_ind_to_xy(proj)
            #    if anotate:
            #        # TODO: Fix position of the labels
            #        self._set_labels(cents, ax, etiquettes, onlyzeros, labelsize, hex=False)
            #else:
            #    #cents = som.bmu_ind_to_xy(np.arange(0, msz[0]*msz[1]))
            #    cents=generate_rect_lattice(msz[0],msz[1])    # modif LB
            #    if anotate:
            #        # TODO: Fix position of the labels
            #        self._set_labels(cents, ax, clusters, onlyzeros, labelsize, hex=False)

            if anotate==True:
                    # TODO: Fix position of the labels
                    cents=coordinates_rectangularRegulargrid(som.codebook.mapsize[0],
                                                             som.codebook.mapsize[1],
                                                             c=.5)
                    #cents = som.bmu_ind_to_xy(np.array(range(0,np.prod(som.mapsize))))
                    #print('cents :\n',cents)
                    self._set_labels(cents,
                                     ax=ax,
                                     labels=labels,
                                     onlyzeros=onlyzeros,
                                     fontsize=labelsize)
        elif som.codebook.lattice == "hexa":
            #raise NotImplementedError('les indices de clusters doivent etre mis en forme.')
            ax.axis('off') # requis a cause de la figure tracée avant
            #print('cluster=\n',clusters)
            ax, cents = plot_hex_map(clusters.flatten()[:,None],
                                     fig=self._fig,
                                     cmap=cmap,
                                     colorbar=False,
                                     titles=[''],
                                     msize=msz)
            #ax, cents = plot_hex_map(clusters.reshape(msz[0], msz[1])[::],  fig=self._fig, colormap=cmap, colorbar=False,titles=['toto'])
            if anotate==True:
                cents, x, y, a=coordinates_hexagonalRegulargrid(som.codebook.mapsize[0],
                                                                som.codebook.mapsize[1],
                                                                r=1/3**.5)
                #print('cents :',cents)
                #print('labels :',clusters.flatten()[::-1])
                self._set_labels(cents,
                                 ax=ax,
                                 labels=labels,
                                 #labels=clusters.flatten()[::-1],
                                 onlyzeros=onlyzeros,
                                 fontsize=labelsize,
                                 hex=True)
            plt.axis('off')
            plt.box(False)
            plt.title(self.title,fontsize=self.text_size)

        else:
            raise ErrorValue('Unexpected lattice : "{}".'.format(som.codebook.lattice))
