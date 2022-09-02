# From Sompy package
# Version 1.0 modifiée LATMOS L. Barthes 28/08/2020

import math

import numpy as np

import matplotlib
import matplotlib.colors
from matplotlib import cm, pyplot as plt

from matplotlib.collections import RegularPolyCollection

from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


from mpl_toolkits.axes_grid1 import make_axes_locatable


def _coordinates_rectangularRegulargrid(x, y,c=.5):
    """
    Old version
    c = demi-coté du carré
    """
    coordinates = [(x-1-2*c*(e%x),2*c*(y-1-e//x)) for e in range(x*y)]
    #print('coordinates:\n',coordinates)
    return coordinates

def coordinates_rectangularRegulargrid(x, y,c=.5):
    """
    c = demi-coté du carré
    """
    offset = 0
    # version hexa # coordinates = [(2*a*(e//x)+a*((e%x)%2),-(a/2+r)*(e%x)) for e in range(x*y)]
    coordinates = [(offset+2*c*(e//x),offset+2*c*(e%x)) for e in range(x*y)]
    #print('coordinates:\n',coordinates)
    return coordinates

def coordinates_hexagonalRegulargrid(x, y,r=1/3**.5):
    """ On créer une grille régulière avec x points par ligne et y points par colonne
        x : nombre de neurones sur la dimension 1 (nombre de neurones par ligne)
        y : nombre de neurones sur la dimension 2 (nombre de neurones par colonne)
        les x sont espacés de r (rayon du cercle circonscrit de l'hexagone). 
        Les y sont espacés de a (apothème de l'hexagone).
    """
    """
        The x coordinates of the map (column numbers) may be thought to range from 0 
        to n-1, where n is the x-dimension of the map, and the y-coordinates (row numbers)
        from 0 to m-1, respectively, where m is the y-dimension of the map. The reference 
        vectors of the map are stored in the map file in the following order :
        1   The unit with coordinates (0  ,0)
        2   The unit with coordinates (1  ,0)
            ...
        n   The unit with coordinates (n-1,0)
        n+1 The unit with coordinates (0  ,1)
            ...

        (0,0)-(1,0)-(2,0)-(3,0)-       (0,0)-(1,0)-(2,0)-(3,0)- 
          |     |     |     |              \ /   \ /   \ /  \  /
        (0,1)-(1,1)-(2,1)-(3,1)-          (0,1)-(1,1)-(2,1)-(3,1)- 
          |     |     |     |              / \   / \   / \   / \    
        (0,2)-(1,2)-(2,2)-(3,2)-       (0,2)-(1,2)-(2,2)-(3,2)- 
          |     |     |     |              \ /   \ /   \ /  \

              rectangular                        hexagonal

        (copier coller de SOM_PAK: 
        The Self-Organizing Map Program Packade (Report A31) de Kohonen et al. )
    """
    a = r * (3**.5 /2) # apothem
    #coordinates = [(2*a*(e%x+.5*((e//y)%2)),-2*a*(e//y)) for e in range(x*y)]
    #coordinates = [(2*a*(e%y+.5*((e//y)%2)),-(a/2+r)*(e//y)) for e in range(x*y)]
    coordinates = [(2*a*(e//x)+a*((e%x)%2),-(a/2+r)*(e%x)) for e in range(x*y)]
    #print('x, y',(x,y))
    #print("coordinates :",coordinates)
    return coordinates, x, y, a

def plot_hex_map(d_matrix, titles=[], cmap=cm.gray, shape=[1, 1], comp_width=5, hex_shrink=1.0, fig=None,
                 colorbar=True,axis=None,msize=None,compsize=8):
    """
    Plot hexagon map where each neuron is represented by a hexagon. The hexagon
    color is given by the distance between the neurons (D-Matrix)

    Args:
    - grid: Grid dictionary (keys: centers, x, y ),
    - d_matrix: array contaning the distances between each neuron
    - w: width of the map in inches
    - title: map title

    Returns the Matplotlib SubAxis instance
    """
    """
    A quoi sert-il de presenter une matrice correspondant à un  tableau de cartes
    """

    # si on n'a pas de reference de figure fournie on la cree
    if fig is None: # code original potentiellement inapproprie (maintenant)
        xinch, yinch = comp_width * shape[1], comp_width * (x / y) * shape[0]
        fig = plt.figure(figsize=(xinch, yinch), dpi=72.)
    
    # on cree les subplot requis
    #print('shape :',shape)
    #print('figure',fig)
    ax_ = fig.subplots(shape[0], shape[1])
    if isinstance(ax_,np.ndarray) and isinstance(ax_[0],np.ndarray):
        ax_ = np.array([ax for sax in ax_ for ax in sax])
        #print('ax_.shape :',ax_.shape)
    # on effectue les transformations de d_matrix pour pouvoir poursuivre
    #d_matrix = np.flip(d_matrix, axis=0)
    #d_matrix = np.flip(d_matrix, axis=1)
    #if d_matrix.ndim < 3:
    #    d_matrix = np.expand_dims(d_matrix, 2)
    
    # on calcule les positions de centres et la taille des hexagones
    r = 1 # rayon circonscrit de l'hexagone -> diametre de 2*r
    r = 1/3**.5
    #print('msize :',msize)
    n_centers, x, y, a = coordinates_hexagonalRegulargrid(*msize,r)
    #print('d_matrix.shape :',d_matrix.shape)
    #print('shape :',shape)
    # on boucle sur chacune des dimensions a afficher
    for comp, title in zip(range(d_matrix.shape[1]), titles):
        if isinstance(ax_,np.ndarray):
            ax = ax_[comp]
        else:
            ax = ax_
        #print('comp :',comp)
        #print('ax :',ax)
        #print('type(ax) :',type(ax))
        ax.axis('off')
        ax.set_title(title,fontsize=compsize)
        #ax.axis('equal')
        #ax.set_ylim((min(ypoints) - 1., max(ypoints) + 1.))
        ax.set_aspect('equal', adjustable='box')
        ax.axis([-a, 2*(y-.5)*a+(y>1)*a, -(x-1)*(r+a/2)-r, r])
        #ax.grid(True)
        #print('xlim :',ax.get_xlim())
        #print('ylim :',ax.get_ylim())

        # les valeurs d'interets par neurone
        d_ = d_matrix[:, comp].squeeze().flatten()
        #d_ = np.flipud(np.fliplr(d_matrix[:,comp].reshape(msize[1], msize[0],order='C')))
        #raise NotImplementedError('A faire.')
    
        # pour la couleur des grandeurs d'interet
        norm = matplotlib.colors.Normalize(vmin=d_.min(), vmax=d_.max())
        facecolors = cmap(norm(d_))
        
        # creation de la collection de polygones
        if False:
            area_inner_circle = math.pi * (apothem ** 2) # ne correspond pas a sizes
            collection_bg = RegularPolyCollection(
                numsides=6,  # a hexagon
                rotation=0,
                sizes=(area_inner_circle,), # en points -> pbm
                facecolors=facecolors,
                #array=d_,
                edgecolors=("black",),
                cmap=cmap,
                norm=norm,
                offsets=n_centers,
                transOffset=ax.transData,
            ) 
        else:
            lesHexagones = []
            for i in range(len(n_centers)) :
                lesHexagones.append(
                    RegularPolygon(
                        xy=n_centers[i],
                        numVertices=6,
                        radius=r,
                        orientation=0.,
                        facecolor=facecolors[i],
                        edgecolor='k'  
                    )
                )
                collection_bg = PatchCollection(lesHexagones, match_original=True)
        # et ajout ...
        ax.add_collection(collection_bg)
        
        # on ajoute la colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # only needed for matplotlib < 3.1

        # colobar horizontale
        if True:
            divider = make_axes_locatable(ax)
            cax = divider.new_horizontal(size="5%", pad=0.5, pack_start=True)
            fig.add_axes(cax)
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        else:
            cbar = fig.colorbar(sm)
            
            
        if not colorbar:
            cbar.remove()
    #cbar.ax.tick_params(labelsize=3 * comp_width)
    if isinstance(ax_,np.ndarray):
        for i in range(d_matrix.shape[1],len(ax_)):
            ax_[i].axis('off')
    return ax_ , list(reversed(n_centers))
