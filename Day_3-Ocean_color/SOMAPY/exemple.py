# From Sompy package
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020

import numpy as np
#import sys
#sys.path.append('./')     # Chemin du package

import matplotlib.pyplot as plt
from sompy import SOMFactory
from sompy import SOMData


#Importation des données

mat = np.loadtxt('el_nino.mat')
Iok  = np.where(mat[:,1] != -9999)[0]
mat = mat[Iok,:]
mat[0,:]= mat[0,:].astype(int)


annee=(mat[:,0]/100).astype(int)
mois=mat[:,0]%100
tmp=np.concatenate((annee[:,np.newaxis], mois[:,np.newaxis]), axis=1)
data=np.concatenate((tmp,mat), axis=1)
comp_names = ['annee','mois','anneemois','SST1','SST2','SST3','SST4','Tx1','Tx2','Tx3','Tx4','Ty1','Ty2','Ty3','Ty4']
masque=np.zeros(len(comp_names))
masque[3]=1               # on n'apprend que sur les 4 SST
masque[4]=1
masque[5]=1
masque[6]=1

data2=data[:,3:7]
data2[0,:] = [0.47,0.06,0.02,0.24]
data3 = data2[300:328,:]
comp_names2 = ['SST1','SST2','SST3','SST4']
masque=None

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list

Inino = get_index_positions(data[:,0],72)
Inino.append(get_index_positions(data[:,0],83))

# Labellisation "Non Nino" ou "Nino"
data_labels = np.empty(len(data2[:,0])).astype(str);
data_labels[0:len(data2[:,0])] = 0; # Initialisation : "Non Nino"
for i in Inino :
    data_labels[i] = 1

#Création d'un objet sData

sData = SOMData(data2, comp_names2, data_labels, normalization = "var")

#Création d'un objet sMap à partir d'un object sData

sm = SOMFactory().build(sData, mapsize=[7,7],normalization = None, initialization='pca', radius_train ='linear', 
               name='El nino',lattice='hexa',mask=masque,components_to_plot=((0,1),(0,2),(0,3)))

#Entraînement de la carte

sm.train(n_job=1, verbose=None, 
         train_rough_len=30, train_rough_radiusin=5,train_rough_radiusfin=1,
         train_finetune_len=50,train_finetune_radiusin=1,train_finetune_radiusfin=0.3,
         watch_evolution = False)

#Labellisation des neurones référents à partir des labels des données

sm.node_labels_from_data(sData.data_labels)

#Affichage des T-SNE

sData.plot_tsne()

sm.plot_tsne()

#Calcul des erreurs globales

topographic_error = sm.calculate_topographic_error()
quantization_error = sm.calculate_quantization_error()
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

#Autres affichages graphiques :

from mapview import View2D
view2D  = View2D(10,10,"El nino",text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)

from bmuhits import BmuHitsView
vhts  = BmuHitsView(10,10,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, logaritmic=False)

from umatrix import UMatrixView
umat = UMatrixView(100,100,"Unified Distance Matrix", text_size=14)
umat.show(sm)

from dendrogram import DendrogramView
dendrogram = DendrogramView(10,10,"Dendrogramme de l'arbre hierarchique", text_size = 10)
dendrogram.show(sm)

from hitmap import HitMapView
sm.cluster(3)
hits  = HitMapView(10,10,"Clustering",text_size=7)
a=hits.show(sm)

plt.show()
