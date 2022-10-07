# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020


from collections import Counter

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm

from mapview import MapView
from plot_tools import coordinates_rectangularRegulargrid
from plot_tools import coordinates_hexagonalRegulargrid
from plot_tools import plot_hex_map
from codebook import generate_rect_lattice, generate_hex_lattice



class BmuHitsView(MapView):
    def _set_labels(self, cents, labels=None, ax=None, onlyzeros=False, fontsize=7, hex=False):
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
            if onlyzeros == True:
                if txt > 0:
                    txt = ""
            #c = cents[i] if hex else (cents[i, 1] + 0.5, cents[-(i + 1), 0] + 0.5)
            #c = (cents[i, 0], cents[i, 1])  # modif LB
            c = (cents[i][0], cents[i][1])  # modif LB
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)

    def show(self, som, anotate=True, onlyzeros=False, labelsize=7, cmap=cm.jet, logaritmic = False):
        if False:
            org_w = self.width
            org_h = self.height
            (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
             axis_num) = self._calculate_figure_params(som, 1, 1)
            self.width /=  (self.width/org_w) if self.width > self.height else (self.height/org_h)
            self.height /=  (self.width / org_w) if self.width > self.height else (self.height / org_h)

        # ajout LB (il faut recalculer les bmu quand on charge un modele)
        som._bmu=som.find_bmu(som._data, njb=1) 
        try: # l'appel de find_bmu peut changer som._bmu alors qu'il ne devrait pas
             # attention, les données peuvent ne pas être normalisées correctement
             # il faut faire attention cela est peut-être dû à la classe mapview
            raise NotImplementedError('Cela casse quelque chose (à reprendre).')
        except NotImplementedError:
            pass # L'idée est la bonne mais il faudra voir ce qui ne va pas.
          
        counts = Counter(som._bmu[0])
        counts = [counts.get(x, 0) for x in range(som.codebook.mapsize[0] * som.codebook.mapsize[1])]
        mp = np.array(counts)
        

        if not logaritmic:
            norm = matplotlib.colors.Normalize(
                vmin=0,
                vmax=np.max(mp.flatten()),
                clip=True)
        else:
            norm = matplotlib.colors.LogNorm(
                vmin=1,
                vmax=np.max(mp.flatten()))

        msz = som.codebook.mapsize

        #cents = som.bmu_ind_to_xy(np.arange(0, msz[0] * msz[1]))
       
        #self.prepare()
        
        self._fig, ax = plt.subplots(1, 1,figsize=(self.width, self.height))
        if som.codebook.lattice == "rect":
            #mp = np.array(counts).reshape(som.codebook.mapsize[1],
            #                              som.codebook.mapsize[0]).T           # modif LB
            #mp = np.flipud(np.fliplr(mp.reshape(msz[1], msz[0],order='C')))
            mp = mp.reshape(msz[0], msz[1],order='F')
            pl=plt.imshow(mp,alpha=1,norm=norm,cmap=cmap)
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
            if anotate:
                cents=coordinates_rectangularRegulargrid(som.codebook.mapsize[0],
                                                         som.codebook.mapsize[1],
                                                         c=.5)
                self._set_labels(cents,
                                 ax=ax,
                                 labels=counts,
                                 onlyzeros=onlyzeros,
                                 fontsize=labelsize)
            #plt.show()
        elif som.codebook.lattice == "hexa":
            #raise NotImplementedError('mp doit etre mis en forme.')
            ax.axis('off')
            #print('mp=',mp,'count=',counts)
            #ax, cents = plot_hex_map(np.flip(mp[::-1],axis=0), cmap=cmap, fig=self._fig,titles=['BMU Hits'])  # ajout LB
            #mp=mp[:,None]
            #print('mp :',mp.shape)
            ax, cents = plot_hex_map(mp[:,None],
                                     cmap=cmap,
                                     fig=self._fig,
                                     titles=['BMU Hits'],
                                     msize=msz)  # ajout LB
            #print('pos=',cents)
            if anotate:
                #print('cents :',cents)
                #print('labels :',mp.flatten()[::-1])
                #self._set_labels(cents, ax, reversed(counts), onlyzeros, labelsize, hex=True)
                self._set_labels(cents,
                                 ax=ax,
                                 labels=mp.flatten()[::-1],
                                 onlyzeros=onlyzeros,
                                 fontsize=labelsize,
                                 hex=True)
            #plt.axis('off')
            #plt.show()
        #return ax, cents
            plt.title(self.title,fontsize=self.text_size)
        else:
            raise ErrorValue('Unexpected lattice : "{}".'.format(som.codebook.lattice))
