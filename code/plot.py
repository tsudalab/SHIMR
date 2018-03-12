import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import itertools
import numpy as np
from calibration import *
import numpy.ma as ma


def plot_DB(predictor, X, y, n_bins, label_pred_all, ir, title, legend_all, label_all, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        point_binary=get_Binary_Features(X, n_bins, point)
        point_fs=predictor.generate_Column_2(point_binary, predictor.feature_set)      
        result.append(predictor._predict(point_fs,predictor.primal_values))
     

    result=np.array(result)    
    result=ir.predict(result.ravel())

    

    Z = np.array(result).reshape(xx.shape)


   
    origin = 'lower'    
    fig=plt.figure(figsize=(8.5,7.5))
    CS=plt.contourf(xx, yy, Z,           
                 cmap=plt.cm.bwr,       
                 levels=[0.4, 0.5, 0.6],
                 extend='both',
                 interpolation='nearest',
                 alpha=0.4, 
                 origin=origin
                 )


    CS2 = plt.contour(CS, levels=CS.levels[::2],
                  colors=('b','r'),
                  origin=origin)

    plt.title(title)

    

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Predicted Probability')
    # # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)

    

    indx_pos=np.where(y==1)[0]
    indx_neg=np.where(y==-1)[0]

    lp=plt.scatter(X[indx_pos][:,0], X[indx_pos][:,1], c='r')
    ln=plt.scatter(X[indx_neg][:,0], X[indx_neg][:,1], c='b')    

    plt.legend((lp,ln),legend_all, scatterpoints=1, loc='upper right', fontsize=8)

   
    plt.xlabel(label_all[0])
    plt.ylabel(label_all[1])
    # plt.title(title)


    reject_indx=np.where(np.array(label_pred_all)==0)[0]
    if(len(reject_indx>0)):        
        plt.scatter(X[reject_indx, 0], X[reject_indx, 1], s=100, facecolors='none', edgecolors='k')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(filename)


def get_Binary_Features(X, n_bins_total, point):    
    X_binary=[]
    for j in range(X.shape[1]): 
        n_bins=n_bins_total 
        while(n_bins>=3):
            X_bins_th=np.linspace(np.min(X[:,j]),np.max(X[:,j]),n_bins)

            for i in range(n_bins-1):
                start=X_bins_th[i]
                if(i==n_bins-2):
                    end=X_bins_th[i+1]
                    if(point[0,j]>=start and point[0,j]<=end):
                        buf=1
                    else:
                        buf=0
                else:
                    end=X_bins_th[i+1]
                    if(point[0,j]>=start and point[0,j]<end):
                        buf=1
                    else:
                        buf=0

                X_binary.append(buf) 

            n_bins-=1 

    return np.array(X_binary).reshape((1,len(X_binary)))


       




