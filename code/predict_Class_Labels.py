import numpy as np


pred_score=[]

def pred_Class(X, Y, predictor):
	for i, X_i in enumerate(X):
		pred_score.append(predictor.predict(X_i))


	pred_score_sorted=np.sort(np.fabs(pred_score))

	for val in pred_score_sorted:
		indx=np.where(np.fabs(pred_score)>val)
		pred_score_filtered=pred_score[indx]
		Y_filtered=Y[indx]
		class_labels=np.sign(pred_score_filtered)











