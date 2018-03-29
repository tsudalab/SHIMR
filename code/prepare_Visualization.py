import numpy as np
import pandas as pd
import math
from visualize import *
import os




def extract_intersection_data(fs, fs_names):
        """
        Extract data to use in intersection bar plot and matrix.
        :return: list of tuples (sets included in intersections), list of tuples (sets excluded from intersections)
        """
        in_sets_list = []
        out_sets_list = []
        
        for _sets in fs:
            in_sets = frozenset(_sets)            
            in_sets_list.append(_sets)
            out_sets = set(fs_names).difference(set(in_sets))            
            out_sets_list.append(set(out_sets))            

        return in_sets_list,  out_sets_list



def display_Rules(fs, rule_list):
	rules_all=[]
	for fs_tmp in fs:
		rule=[]		
		for item in fs_tmp:
			indx=int(item)
			rule.append(rule_list[indx])
		rules_all.append(rule)
	print(rules_all)




def visualize_main(n_bins,d,predictor,filename_rules,Feature_dict,data_test,f_Rules_Actual_with_Diag,dir_figure,class_label_dict,f_feature_range_array,f_rule_array,pt_disp,plot_all):
	
	### Load actual rules with diagnosis ###
	data=pd.read_csv(f_Rules_Actual_with_Diag)
	rules_actual_per_RID=data.as_matrix()

	### Load feature range array
	feature_range_array=np.load(f_feature_range_array)

	### Load rule array
	rule_array=np.load(f_rule_array)	

	### Load rules from file
	rule_list_data=pd.read_csv(filename_rules, header=None)
	rule_list_mat=rule_list_data.as_matrix()
	rule_list = rule_list_mat.ravel().tolist()


	primal_values=predictor.primal_values
	bias_value1=predictor._bias
	bias_value2=(-1)*np.sum(predictor.primal_values) ### considering (0-1) model
	bias_value=bias_value1 + bias_value2



	# Fetch the feature names	
	f_names=Feature_dict.keys() # Actual attribute names
	f_names_list=[]
	for key in f_names:
		f_names_list.append(key)	

	
	fs_selected_all = predictor.feature_set 	# Selected features (by the model)
	

	indx_not_zero=np.where(primal_values!=0)[0] # Find the non zero weight indices of the selected features
	fs_selected=np.array(fs_selected_all)[indx_not_zero]	
	# fs_wt=primal_values[indx_not_zero]
	fs_wt=2*primal_values[indx_not_zero] ### considering (0-1) model


	fs=fs_selected
	wt=fs_wt


	### sort the rules based on decreasing weights
	indx_sorted=np.argsort(np.absolute(wt))[::-1]
	fs=fs[indx_sorted][0:10] # top 10 rules
	wt=wt[indx_sorted][0:10] # top 10 rules	

	# fs=fs[indx_sorted] # all selected rules
	# wt=wt[indx_sorted] # all selected rules	

	if(pt_disp):
		display_Rules(fs, rule_list)
		print(wt)



	score_min=min(rules_actual_per_RID[:,3])
	score_max=max(rules_actual_per_RID[:,3])
	score_range=(score_min - 1,score_max + 1)


	for i in range(rules_actual_per_RID.shape[0]):
		RID=rules_actual_per_RID[i,0]		

		bar_plot_file_path= dir_figure + 'figure_RID_' + str(RID) + '.pdf'	

		indx_X_test=np.where(data_test[:,0]==RID)[0]
		val_X_test=data_test[indx_X_test,2:]

		diag_pred=class_label_dict[rules_actual_per_RID[i, 2]]
		model_score=rules_actual_per_RID[i, 3]

		fs_RID=[] # holds rule indices of a particular 'RID'
		_is=rules_actual_per_RID[i, 5].strip('[,]').replace(" ", "").split(',')	
		for _item in _is:
			fs_RID.append(int(math.floor(int(_item)/(n_bins-1))))

		### Create itemsets based on row matrix indices ("intersection matrix")
		indx_set_all=[]
		indx_set_matched_all=[] # Needed to draw a blue rectangle around the selected items
		indx_set_matched_X_val=[] # Needed to calculate the relative x-position for drawing a blue rectangle around the selected items
		item_set_RID=[]
		wt_all=[]
	
		count=0
		for fs_tmp in fs:	# loop through the "selected features by the model". Each feature contains a set of rules.
			indx_set=[]	# holds feature index of each rule of a selected itemsets (features) by the model
			for item in fs_tmp:	
				item=str(item).strip(' ')		
				_indx=int(math.floor(int(item)/(n_bins-1)))
				indx_set.append(_indx)

			if(np.intersect1d(indx_set,fs_RID).shape[0]>0): # consider only those "model itemsets" which contains atleast one individual feature.
				
				if(np.intersect1d(fs_tmp,_is).shape[0]==len(fs_tmp)): # Needed to draw a blue rectangle around the selected items
					indx_set_matched_all.append(indx_set)
					indx_set_matched_X_val.append(len(indx_set_all)) # len(indx_set_all) represents the relative x-position in intersection matrix

				indx_set_all.append(indx_set) # feature index
				wt_all.append(wt[count])
				item_set_RID.append(fs_tmp) # item index
			count+=1

		
		### Create fs_names list acoording to the weights ###
		label_all_indx=[]
		for fs_tmp in indx_set_all:
			for item in fs_tmp:
				if item not in label_all_indx:
					label_all_indx.append(item)
		in_sets_list, out_sets_list= extract_intersection_data(indx_set_all, label_all_indx)

		fs_names=np.array(f_names_list)[label_all_indx]
		set_row_map = dict(zip(label_all_indx, np.arange(len(label_all_indx))+1))



		rows=len(label_all_indx) # Number of rows
		cols=len(wt_all)		 # Number of columns
		title='Selected rules \n(' + r"$\bf{" + str(diag_pred) + "}$"  + ') \n Model Score =  {0:0.02f}'.format(model_score) + ', bias= {0:0.2f}'.format(bias_value) 
		
		fs_plot=FS_Plot(rows,cols,in_sets_list,out_sets_list,np.array(wt_all),fs_names,bar_plot_file_path,set_row_map,fs,item_set_RID,n_bins,title,val_X_test,indx_set_matched_all,indx_set_matched_X_val,score_range,model_score,feature_range_array,rule_array,pt_disp)

		print('RID: ' + str(RID) + ' done ...')

		if not plot_all: # If visualization of all 'RIDs' are required, then set "plot_all=True".
			break







































