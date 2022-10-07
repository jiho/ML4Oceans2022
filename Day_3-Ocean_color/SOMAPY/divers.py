import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# fonction d'affichage des matrices de confusion (copié/collé du site de scikit-learn)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fontsize=16):
    """
    This function printed and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Attention : les classes commencent à zero
    copier/coller d'un tutoriel sklearn?
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # suppose que les classes sont numerotees à partir de 0
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [ classes[i] for i in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    #fig, ax = plt.subplots()
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes
           #title=title,
           #ylabel='True label',
           #xlabel='Predicted label'
          )
    ax.set_title(title,fontsize=fontsize)
    ax.set_xlabel('Predicted label',fontsize=fontsize)
    ax.set_xticklabels(classes,fontsize=fontsize)
    ax.set_ylabel('True label',fontsize=fontsize)
    ax.set_yticklabels(classes,fontsize=fontsize)
    
    ## Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor",fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",fontsize=fontsize,
                    color="white" if cm[i, j] > thresh else "black")
    return ax
