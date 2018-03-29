import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def grid_search_LR(X,y, filename, path_param):
	# Now use classifier
	C_range = np.logspace(-3, 2, 50)
	param_grid = dict(C=C_range)
	cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.3, random_state=42) ### 10-Fold
	grid = GridSearchCV(LogisticRegression(solver='lbfgs', class_weight='balanced'), param_grid=param_grid, cv=cv, scoring='roc_auc')	
	grid.fit(X.astype(float), y.astype(float))

	# File write the Grid Search Cross Validation Results.
	scores = [x[1] for x in grid.grid_scores_]
	np.savetxt(path_param+'scores_GS_'+ filename, np.array(scores).T) 
	np.savetxt(path_param+'C_range_'+filename, C_range.reshape((C_range.shape[0],1)))


	print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))

	np.savetxt(path_param+'Best_Params_Scores_'+filename,[grid.best_params_['C'],grid.best_score_]) 

	C_best, score_best=np.loadtxt(path_param+'Best_Params_Scores_'+filename)
	print("The best parameters are %s with a score of %0.2f"% (C_best, score_best))



	print('Done !')



	