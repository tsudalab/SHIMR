import csv
import sys
import os
import shutil
from load_WDBC_Data import *
from binarize_Data import *
from reject_boost import *
from filewrite_Actual_Rules_Per_RID import *
from prepare_Visualization import *


data_path="../Data/tmp/"
image_path="../Images/"
DB_path="../Images/DB/"
fname ='input_data_SHIMR_wdbc'


diag_dict2={'B':'Benign', 'M':'Malignant'}
diag_dict3={-1:'Benign', +1:'Malignant', 0:'Rejected'}


def create_Dirs(dir_paths):
	for loc_path in dir_paths:
		if not os.path.exists(loc_path):
			os.makedirs(loc_path)




def run(data,d=0.5,n_bins=7,C_POS=1,C_NEG=1,size_U=3,apply_rejection=False,plot_all=False):	
	X_train,Y_train,X_test,Y_test, Feature_dict=data

	#Binarize data
	X_train_binarized,X_test_binarized=binarize(X_train, X_test, n_bins, Feature_dict, data_path)


	while(d>0):		

		rb=reject_boost(Y_train, d, n_bins, size_U, data_path, fname)	
		predictor=rb.solve_Dual(X_train_binarized, Y_train, d, C_POS, C_NEG)

		
		f_primal=data_path + 'primal_values.npy'
		f_bias=data_path + 'bias_value_d_' + str(d) + '.npy'
		f_bias2=data_path + 'bias_value_2_d_' + str(d) + '.npy'
		filename_rules= data_path + 'Rule_List_' + 'nbins_'+ str(n_bins) + '.csv'	
		f_feature_set=data_path + 'feature_set_d_' + str(d) + '.csv'
		f_X_text=data_path + 'X_test.npy'
		f_RID_vs_Diag_test= data_path + 'RID_vs_Diag_test.npy'
		f_Rules_Actual_with_Diag_test= data_path + 'actual_Rules_with_Diag_Per_RID_Test_' + str(d) +'.csv'
		f_itemset=data_path+'itemset_test.csv'

		dir_figure=image_path + 'Interaction_Plot_d_' + str(d) + '/'
		if not os.path.exists(dir_figure):
			os.makedirs(dir_figure)


		

		# Solve the primal	
		if(predictor.feature_set):		
			rb.solve_Primal(predictor, X_train_binarized, C_POS, C_NEG)

			

			### Write selected features into file
			csv_fs=open(f_feature_set, 'w')
			fs_writer=csv.writer(csv_fs)
			fs_writer.writerows(predictor.feature_set)
			csv_fs.close()

			# Save feature weights
			np.save(f_primal,predictor.primal_values)
			np.save(f_bias,predictor._bias)
			np.save(f_bias2,(-1)*np.sum(predictor.primal_values))  ### considering (0-1) model

		

			
			# Apply model on 'Training Data'	
			rb.apply_model(predictor, X_train_binarized, Y_train, FLAG=1)

			# Apply model on 'Test Data'
			rb.apply_model(predictor, X_test_binarized, Y_test, FLAG=0)

			write_Actual_Rules_with_Diag_per_RID(f_Rules_Actual_with_Diag_test, X_test_binarized, f_RID_vs_Diag_test, filename_rules, rb.result_train, rb.pred_proba_cal, rb.label_pred_all,diag_dict2,diag_dict3,f_itemset)



			visualize_main(n_bins,d,f_primal,f_bias,f_bias2,filename_rules,Feature_dict,f_feature_set,X_test,f_RID_vs_Diag_test,f_Rules_Actual_with_Diag_test,data_path,dir_figure,plot_all)

			# ### Plot Decision Boundary of Test Data ###
			# f_name=image_path+'DB/Moon/' + 'Decision_Boundary_Test_n_bins_' + str(n_bins)+ '_' +  fname + '_dual_boost_L1_d_' + str(d) + '.pdf'		
			# title='DB vs RR (C1 vs C2) \n (rr= ' +str(rb.rr) +  ', acc= ' +str(rb.acc) + ', auc= '+str(rb.roc_auc) +')'
			# legend_all=('C1','C2')
			# label_all=['F1', 'F2']
			# plot_DB(predictor, X_test_orig, Y_test, n_bins, rb.label_pred_all, rb.ir, title, legend_all, label_all, 50, f_name)

			
		else:
			print('Feature set is empty!!!')

		if(apply_rejection):
			d=round((d-0.005),3)
			if(d==0.49):
				break
		else:
			break

	### Delete the temporary files ###
	if os.path.exists(data_path):
		shutil.rmtree(data_path)





if __name__=="__main__":
	create_Dirs([data_path,image_path])
	data=load_data(data_path)

	if(len(sys.argv)==2):
		run(data,size_U=int(sys.argv[1]))
	else:
		run(data)

	print('Done...')
