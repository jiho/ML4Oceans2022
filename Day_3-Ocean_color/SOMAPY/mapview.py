# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020

import matplotlib
from matplotlib import colors
from matplotlib import cm
from matplotlib import pyplot as plt

from copy import deepcopy, copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from sompy.visualization.plot_tools import plot_hex_map
from plot_tools import coordinates_rectangularRegulargrid
from plot_tools import coordinates_hexagonalRegulargrid
from plot_tools import plot_hex_map
#from .view import MatplotView
from view import MatplotView
import numpy as np


class MapView(MatplotView):

    def _calculate_figure_params(self, som, which_dim, col_sz):

        try :
            codebook = np.asarray(copy(som.codebook.matrix))
            # add this to avoid error when normalization is not used
            if som._normalizer and type(som._normalizer) != list:
                #codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
                codebook = som._normalizer.denormalize(copy(codebook)) # modif LB
                raise NotImplementedError('')
        except NotImplementedError:
            pass # présent aussi dans le show
                 # il faudra sûrement faire un choix
                 
        indtoshow, sV, sH = None, None, None

        if isinstance(which_dim,str) and which_dim == 'all':
            dim = som._dim
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.arange(0, dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap

        elif isinstance(which_dim,int):
            dim = 1
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1).astype(int)
            indtoshow[0] = int(which_dim)
            sH, sV = 16, 16 * ratio_hitmap

        elif isinstance(which_dim,(list,np.ndarray)):
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap

        no_row_in_plot = max(1,dim // col_sz)  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)


class View2D(MapView):

    # method copy/past (ac)
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
            if onlyzeros == True and txt !='0':
                txt = ""
            c = (cents[i][0], cents[i][1])  # modif LB
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)
            #print('i,txt (c) :',i,txt,c)
            
    def show(self, som,what='codebook', which_dim='all',col_sz=None, denormalize=False,anotate=False,neuronLabels=None,onlyzeros=False, labelsize=7, cmap=cm.jet,compsize=8):
        (_, _, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz) # compliquer a virer
        # on n'utilise pas axis_num
        msz = som.codebook.mapsize
        codebook = np.asarray(copy(som.codebook.matrix))      
        if denormalize:
            try:
                for i in range(len(som._normalizer)):
                    codebook[:,i] = np.asarray(som._normalizer[i].denormalize(som.codebook.matrix[:,i]))
                raise NotImplementedError('')
            except NotImplementedError:
                pass # présent aussi dans le _calculate_figure_params
                 # il faudra sûrement faire un choix

            
        if False:
            if which_dim == 'all':
                names = som._component_names[0]
            elif type(which_dim) == int:
                names = [som._component_names[0][which_dim]]
                codebook = codebook[:,which_dim][:,None]    # ajout LB/ac
            elif isinstance(which_dim,(list,np.ndarray,range)): # or type(which_dim) == range:
                names = som._component_names[0][which_dim[0]:which_dim[-1]+1]                # Modif LB
                codebook=codebook[:,which_dim[0]:which_dim[-1]+1]                             # ajout LB
        else:
            # on limite les variables
            # (noms et colonnes de codebook) 
            #print('indtoshow :',indtoshow)
            codebook = codebook[:,indtoshow]
            names = [som._component_names[0][i] for i in indtoshow]
            no_row_in_plot = np.ceil(codebook.shape[1]/no_col_in_plot).astype(int)
        #plt.clf() ; 
        
        #indtoshow = range(len(names))
        if som.codebook.lattice=="rect":
            #print(codebook)
            #print("no_row_in_plot, no_col_in_plot : ",no_row_in_plot, no_col_in_plot)
            self._fig, ax_ = plt.subplots(no_row_in_plot,
                                          no_col_in_plot,
                                          figsize=(self.width, self.height))
            if isinstance(ax_,np.ndarray) and isinstance(ax_[0],np.ndarray):
                ax_ = np.array([ax for sax in ax_ for ax in sax])
                
            #while axis_num < len(indtoshow):
            for ind in range(codebook.shape[1]):
                if codebook.shape[1] == 1:
                    ax = ax_
                else:
                    ax = ax_[ind]#plt.subplot(, axis_num,figsize=(self.width,self.height))
                #ind = int(indtoshow[axis_num])
                #print('ind :',ind)
                min_color_scale = min(codebook[:, ind].flatten())               # ajout LB
                max_color_scale = max(codebook[:, ind].flatten())               # ajout LB

                #print(min_color_scale,max_color_scale)
                norm = matplotlib.colors.Normalize(vmin=min_color_scale, vmax=max_color_scale, clip=True)
                # pipo = np.flipud(np.fliplr(codebook[:,ind].reshape(msz[1], msz[0],order='C')))
                pipo = codebook[:,ind].reshape(msz[1], msz[0],order='C')
                #print('pipo\n',pipo)
                pl = ax.imshow(codebook[:,ind].reshape(msz[0], msz[1],order='F'), alpha=1, norm=norm,cmap=cmap)
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
                
                # on affiche le titre
                ax.set_title(names[ind],fontsize=self.text_size)
                
                # colobar horizontale
                if True:
                    divider = make_axes_locatable(ax)
                    cax = divider.new_vertical(size="10%", pad=0.5, pack_start=True)
                    self._fig.add_axes(cax)
                    self._fig.colorbar(pl, cax=cax, orientation="horizontal")
                    #plt.colorbar(pl,fraction=.2,aspect=30, orientation="horizontal")
                    ##plt.colorbar(pl)#,aspect=100, orientation="horizontal")
                else:
                    divider = make_axes_locatable(ax)
                    cax = divider.new_horizontal(size="10%", pad=0.5, pack_start=False)
                    self._fig.add_axes(cax)
                    self._fig.colorbar(pl, cax=cax, orientation="vertical")
                    
                #if neuronLabels != None and anotate==True:
                if anotate==True:
                    # TODO: Fix position of the labels
                    cents=coordinates_rectangularRegulargrid(som.codebook.mapsize[0],
                                                             som.codebook.mapsize[1],
                                                             c=.5)
                    #cents = som.bmu_ind_to_xy(np.array(range(0,np.prod(som.mapsize))))
                    self._set_labels(cents,
                                     ax=ax,
                                     labels=neuronLabels,
                                     onlyzeros=onlyzeros,
                                     fontsize=labelsize)

            if isinstance(ax_,np.ndarray):
                for i in range(codebook.shape[1],len(ax_)):
                    ax_[i].axis('off')
            #axis_num += 1 # fin du while (boucle sur les dimensions)
                

                    

        elif som.codebook.lattice=="hexa":
            #print('indtoshow :',indtoshow)
            #print('codebook.shape :',codebook.shape)
            #codebook = codebook[:,indtoshow]
            # affichage de la carte hexa
            #print('codebook.shape :',codebook.shape)
            #print('names :',names)
            #print('indtoshow :',indtoshow)
            #names = [names[i] for i in indtoshow]
            #print('codebook.shape :',codebook.shape)
            #print('figure')
            #self._fig = plt.figure(figsize=(self.width, self.height))
            fig = plt.figure(figsize=(self.width, self.height))
            #print('som.codebook.mapsize :',som.codebook.mapsize)
            #print('codebook :\n',codebook)
            #tmp=np.transpose(codebook.reshape((som.codebook.mapsize[1],
            #                                   som.codebook.mapsize[0],
            #                                   codebook.shape[-1])),
            #                 axes=(1,0,2))  # ajout LB
            #no_row_in_plot = np.ceil(codebook.shape[1]/no_col_in_plot).astype(int)
            ax, cents = plot_hex_map(codebook,
                                     titles=names,    # modif LB
                                     shape=[no_row_in_plot, no_col_in_plot],
                                     cmap=cmap,
                                     fig=fig,
                                     msize=msz,compsize=compsize)

            #if neuronLabels != None and anotate==True:
            if anotate==True:
                cents, x, y, a=coordinates_hexagonalRegulargrid(som.codebook.mapsize[0],
                                                                som.codebook.mapsize[1],
                                                                r=1/3**.5)
                #if codebook.shape[-1]==1:
                #    self._set_labels(cents,
                #                     ax=ax,
                #                     labels=neuronLabels,
                #                     onlyzeros=onlyzeros,
                #                     fontsize=labelsize,
                #                     hex=True)
                #else:
                #    for i in range(codebook.shape[-1]):
                #        self._set_labels(cents,
                #                         ax=ax[i],
                #                         labels=neuronLabels,
                #                         onlyzeros=onlyzeros,
                #                         fontsize=labelsize,
                #                         hex=True)

                if isinstance(ax,np.ndarray):
                    for ax_i in ax:
                        self._set_labels(cents,
                                         ax=ax_i,
                                         labels=neuronLabels,
                                         onlyzeros=onlyzeros,
                                         fontsize=labelsize,
                                         hex=True)
                else:
                    self._set_labels(cents,
                                     ax=ax,
                                     labels=neuronLabels,
                                     onlyzeros=onlyzeros,
                                     fontsize=labelsize,
                                     hex=True)



class View2DPacked(MapView):

    def _set_axis(self, ax, msz0, msz1):
        plt.axis([0, msz0, 0, msz1])
        plt.axis('off')
        ax.axis('off')

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        if col_sz is None:
            col_sz = 6
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        codebook = som.codebook.matrix

        cmap = cmap or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize
        compname = som._component_names
        if what == 'codebook':
            h = .1
            w = .1
            self.width = no_col_in_plot*2.5*(1+w)
            self.height = no_row_in_plot*2.5*(1+h)
            self.prepare()

            while axis_num < len(indtoshow):
                axis_num += 1
                ax = self._fig.add_subplot(no_row_in_plot, no_col_in_plot,
                                           axis_num)
                ax.axis('off')
                ind = int(indtoshow[axis_num-1])
                mp = codebook[:, ind].reshape(msz0, msz1)
                plt.imshow(mp[::-1], norm=None, cmap=cmap)
                self._set_axis(ax, msz0, msz1)

                if self.show_text is True:
                    plt.title(compname[0][ind])
                    font = {'size': self.text_size}
                    plt.rc('font', **font)
        if what == 'cluster':
            try:
                codebook = getattr(som, 'cluster_labels')
            except:
                codebook = som.cluster()

            h = .2
            w = .001
            self.width = msz0/2
            self.height = msz1/2
            self.prepare()

            ax = self._fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            plt.imshow(mp[::-1], cmap=cmap)

            self._set_axis(ax, msz0, msz1)

        plt.subplots_adjust(hspace=h, wspace=w)

        #plt.show()
        
        
class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()

        codebook = som.codebook.matrix

        while axis_num < len(indtoshow):
            axis_num += 1
            plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)

        #plt.show()


