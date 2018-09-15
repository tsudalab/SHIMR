#####################################################
import numpy as np



def get_Classification_Results(label_pred_all,y):
	rejected=0
	rejected_pos=0
	rejected_neg=0
	correctly_classified=0
	misclassified=0
	pos_correctly_classified=0
	neg_correctly_classified=0
	pos_misclassified=0
	neg_misclassified=0
	indx_misclassified=[]
	
	for i in range(y.shape[0]):
		if(label_pred_all[i]==0):
			rejected+=1

			if(y[i]==+1):
				rejected_pos+=1
			else:
				rejected_neg+=1
		else:
			if(label_pred_all[i]==y[i]):
				correctly_classified+=1
				if(y[i]==+1):
					pos_correctly_classified+=1
				else:
					neg_correctly_classified+=1
				
			else:
				misclassified+=1
				indx_misclassified.append([i])
				if(y[i]==+1):
					pos_misclassified+=1
				else:
					neg_misclassified+=1
	
	return [correctly_classified, misclassified, rejected, rejected_pos, rejected_neg, pos_correctly_classified, neg_correctly_classified, pos_misclassified, neg_misclassified ]
