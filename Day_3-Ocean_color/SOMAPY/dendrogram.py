# From Sompy package
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020

from view import MatplotView
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

class DendrogramView(MatplotView):
    
    def show(self, som,labels=None,leaf_font_size=None,n_color_clusters=None,method='ward', metric='euclidean', optimal_ordering=False):
        
        self.prepare()

            
        if False: # ce qu'il y avait avant
            Z = linkage(np.triu(som._distance_matrix),'ward','euclidean');
        else:
            if som.mask is None:
                Z = linkage(som.codebook.matrix[:,:],
                            method=method,
                            metric=metric,
                            optimal_ordering=optimal_ordering)
            else : 
                selec_mask = np.where(som.mask==1)[0]  
                Z = linkage(som.codebook.matrix[:,selec_mask],
                            method=method,
                            metric=metric,
                            optimal_ordering=optimal_ordering)

        # ajoute pour prendre en compte les couleurs d'un nombre de clusters
        color_threshold = None
        if n_color_clusters is not None:
            if isinstance(n_color_clusters,(int)):
                # calculate color threshold
                color_threshold = Z[-(n_color_clusters-1),2]  
            else:
                raise TypeError('Unexpected')
            
        if False: # ce qu'il y avait avant
            dendrogram(Z,som.codebook.nnodes,'lastp');
        else:
            plt.axis('on')
            dendrogram(Z,som.codebook.nnodes,'lastp',show_leaf_counts=True,labels=labels,leaf_font_size=leaf_font_size,color_threshold=color_threshold)
        #plt.show()

        if True :#and color_threshold is not None:
            plt.gca().plot(plt.gca().get_xlim(),[color_threshold, color_threshold],'k--')
        #return 1

