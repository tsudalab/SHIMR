import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import label_binarize
from grid_search_LR import *



def get_Probability(prob_pos, y, filename):
	fig = plt.figure(1, figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

	prob_pos_SVM = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	brier_score = brier_score_loss(y, prob_pos_SVM, pos_label=y.max())
	fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos_SVM, n_bins=10)	
	ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='%s, (%1.3f)'%('SVM O/P',brier_score))


	lr=get_Sigmoid_Calibrated_Probability(prob_pos,y)
	prob_pos_Sigmoid = lr.predict_proba(prob_pos)[:, 1]
	brier_score = brier_score_loss(y, prob_pos_Sigmoid, pos_label=y.max())
	fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos_Sigmoid, n_bins=10)
	ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='%s, (%1.3f)'%('Sigmoid Calibrated',brier_score))

	ir=get_Isotonic_Calibrated_Probability(prob_pos,y)
	prob_pos_Isotonic=ir.predict(prob_pos.ravel())
	brier_score = brier_score_loss(y, prob_pos_Isotonic, pos_label=y.max())
	fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos_Isotonic, n_bins=10)
	ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='%s, (%1.3f)'%('Isotonic Calibrated',brier_score))

	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')
	plt.savefig('../Images/'+filename)


def get_Sigmoid_Calibrated_Probability(X,Y):
	X=X.reshape((X.shape[0],1))
	C_best=do_Grid_Search(X,Y)
	lr = LogisticRegression(C=C_best, solver='lbfgs')
	lr.fit(X, Y)
	# prob_pos = lr.predict_proba(X)[:, 1]
	return lr

def get_Isotonic_Calibrated_Probability(X,Y):
	X=X.ravel()
	ir = IsotonicRegression(out_of_bounds = 'clip')
	Y=label_binarize(Y.tolist(), classes=[-1,1]).ravel()
	# prob_pos = ir.fit_transform(X, Y)
	# return prob_pos
	prob_pos = ir.fit(X, Y)
	return ir

def do_Grid_Search(X,Y):
	path_param='./'
	## Do the grid search
	file_name_GS='LR_GS.txt'
	grid_search_LR(X.astype(float), Y, file_name_GS, path_param)

	
	C_best, score_best=np.loadtxt(path_param+'Best_Params_Scores_'+ file_name_GS)
	print("The best parameters are C= %f, with a score (AUC) of %0.2f"% (C_best, score_best))
	
	return C_best







    