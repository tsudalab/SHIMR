#####################################################
from __future__ import division
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from reject_boost_train import *
from reject_boost_dual_optimizer_input import *
from reject_boost_primal_optimizer_input import *
from plot import *
from predict_Class_Labels import *
from calibration import *
from classification_Results import *








class reject_boost(object):
	def __init__(self, d, size_U, C_POS, C_NEG, data_path, fname):		
		self.d=d
		# self.n_bins=n_bins	
		self.size_U=size_U
		self.C_POS=C_POS
		self.C_NEG=C_NEG
		self.ir=None
		self.filename_dual=fname + '_dual_boost_L1_d_' + str(d) + '.lp'
		self.filename_primal=fname + '_primal_iBoost_L1_d_' + str(d) + '.lp'
		self.data_path=data_path	

		

	def solve_Dual(self, X, Y):
		self.Opt_dual=Optimizer(Y, self.d)
		### Create an instance of "Reject_Boost_Trainer"
		trainer = Reject_Boost_Trainer(self.data_path,self.size_U)
		predictor = trainer.train(X, Y, self.d, self.C_POS, self.C_NEG, self.Opt_dual, self.filename_dual)	
		return predictor


	def solve_Primal(self, predictor, X, Y):
		self.Opt_Primal=Optimizer_Primal(Y, self.d)
		buf_all=predictor.generate_Column_2(X, predictor.feature_set)		
		predictor._solve_LP_Primal(self.Opt_Primal, buf_all, self.C_POS, self.C_NEG, self.filename_primal)


	def predict(self, predictor, buf_all):
		result_train=[]	
		for i, X_i in enumerate(buf_all):			
		    pred=predictor._predict(X_i, predictor.primal_values)			    
		    result_train.append(pred)
		
		result_train=np.array(result_train)		
		result_train=result_train.reshape((result_train.shape[0],1))
		return result_train

	def apply_model(self, predictor, X, Y,  FLAG):		
		buf_all=predictor.generate_Column_2(X, predictor.feature_set)			

		if(np.any(np.absolute(predictor.primal_values)) == False):
			print('All multipliers are zero !')
		else:		
			result_train=self.predict(predictor, buf_all)				

			if(FLAG==0 and self.ir is None): # FLAG=0 => 'Testing'
				print('Please do the training first !!!')
				return
			elif(FLAG==1 and self.ir is None): # FLAG=1 => 'Training'
				ir=get_Isotonic_Calibrated_Probability(np.array(result_train),Y)
				self.ir=ir

			pred_proba_cal=self.ir.predict(result_train.ravel())

			

			label_pred_all=[]
			for i in range(pred_proba_cal.shape[0]):
				if(pred_proba_cal[i]>1-self.d):
					label_pred=+1
				elif(pred_proba_cal[i]<self.d):
					label_pred=-1
				else:
				    label_pred=0

				label_pred_all.append(label_pred) 

		

			[correctly_classified, misclassified, rejected, rejected_pos, rejected_neg, TP, TN, FN, FP ] = \
			get_Classification_Results(label_pred_all,Y)

			SN=round(TP/(TP+FN),2)
			SP=round(TN/(TN+FP),2)
			PR=round(TP/(TP+FP),2)

			indx_primal=np.where(predictor.primal_values!=0)[0]
			rule_size=len(indx_primal)

			if(FLAG):
				mode='Training'
			else:
				mode='Testing'
			print("======= " + mode +" Results =======")
			print('d=' + str(self.d))
			print('No of rules selected = ' +str(rule_size))
			print('correctly_classified:' + str(correctly_classified) + ', misclassified: '+ str(misclassified) + ', rejected: '+str(rejected))
			print('TP:'+str(TP)+', TN: '+str(TN)+', FP:'+str(FP)+', FN: '+str(FN)+', SN/RC:'+str(SN)+', PR:'+str(PR)+', SP: '+str(SP))

			indx=np.where(np.array(label_pred_all)!=0)[0]
			if(len(indx)>0):				
				roc_auc=''
				if(np.unique(Y[indx]).shape[0]>1):						
					fpr, tpr, _ = roc_curve(Y[indx].tolist(), pred_proba_cal[indx])
					roc_auc = round(auc(fpr, tpr),2)
					print('roc_auc: '+ str(roc_auc))
					area_pr=round(average_precision_score(Y[indx].tolist(), pred_proba_cal[indx]),2)
					print('area_pr: '+ str(area_pr))

				rr=round(((Y.shape[0] - len(indx) ) / Y.shape[0]),2)
				acc=round(accuracy_score(Y[indx].tolist(),np.array(label_pred_all)[indx]),2)		
				print('accuracy: '+ str(acc))
				print('rejection rate: '+ str(rr))			
				
				self.label_pred_all=label_pred_all	
				self.result_train=result_train
				self.pred_proba_cal=pred_proba_cal
				self.rr=rr
				self.acc=acc
				self.roc_auc=roc_auc			
			else:
				print('All data points are rejected (rejection rate = 100 %)')

	


	def fit(self, X_train_binarized, Y):
		predictor=self.solve_Dual(X_train_binarized, Y)
		self.predictor=predictor
		if(predictor.feature_set):
			self.solve_Primal(predictor, X_train_binarized, Y)

			# Apply model on 'Training Data'	
			self.apply_model(predictor, X_train_binarized, Y, FLAG=1)
		else:
			print('Feature set is empty!!!')	

	def test(self, X_test_binarized, Y_test):
		# Apply model on 'Test Data'
		if(self.predictor.feature_set):
			self.apply_model(self.predictor, X_test_binarized, Y_test, FLAG=0)
		else:
			print('Feature set is empty!!!')



















