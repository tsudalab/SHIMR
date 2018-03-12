import numpy as np
import pandas as pd
import csv
import os
import collections






### Given the binary data, this script generates itemset file. This will be used as input to "lcm" 
def generate_Itemset(X, filename):
    # filename="ADNI.txt"
    if os.path.exists(filename):
        os.remove(filename)
    file_fs=open(filename, "a+")  
    indx_transaction=[]  
    for i in range(X.shape[0]):
        indx=np.where(X[i,:]==1)[0].tolist()
        if len(indx)>0:
            indx_transaction.append(i)
            count=0
            for item in indx:
            	if(count>0):
            		file_fs.write(' ')
            	file_fs.write('%d' %item)
            	count+=1
                 
            file_fs.write('\n')
        else:
        	print('No transaction for this RID')     
    file_fs.close()
    return indx_transaction





def write_Actual_Rules_per_RID(f_Actual_Rules_Per_RID, X, f_RID_Diag, filename_rules,diag_dict2,filename_IS):
	RID_vs_Diag=np.load(f_RID_Diag)
	RID=RID_vs_Diag[:,0]
	Diag=RID_vs_Diag[:,1]

	


	### load the rules
	rule_list_data=pd.read_csv(filename_rules, header=None)
	rule_list_mat=rule_list_data.as_matrix()
	rule_list = rule_list_mat.ravel().tolist()
	

	# Generate itemset corresponding to each RID and then file write and load it	
	generate_Itemset(X, filename_IS)
	data_df=pd.read_csv(filename_IS, sep=' ', header=None)
	data=data_df.as_matrix() # 'data' contains itemsets corresponding to each 'RID'
	N=data.shape[0] # Number of RIDs


	file_csv=open(f_Actual_Rules_Per_RID, 'w')
	file_wr=csv.writer(file_csv)
	header_string=['RID', 'diag_act', 'itemsets',  'rules']
	file_wr.writerow(header_string) # Write the file header

	# Dict_RID_FS=collections.OrderedDict() # Dictionary to contain all the rules associated with each 'RID'
	for i in range(N):
		tmp=data[i][pd.notnull(data[i])].astype(np.int32)		
		diag_act=diag_dict2[Diag[i]]
		
		
		itemsets=[]
		rules_all=[]		
			
		for item in tmp:			
			itemsets.append(item)		
			rule=rule_list[int(item)]	
			rules_all.append([rule])
	

	
		buf=[RID[i], diag_act, itemsets, rules_all]			
		file_wr.writerow(buf)

	file_csv.close()
	
			



def write_Actual_Rules_with_Diag_per_RID(f_Actual_Rules_Per_RID, X, f_RID_Diag, filename_rules, result_train, pred_proba_cal, label_pred_all,diag_dict2,diag_dict3,filename_IS):
	RID_vs_Diag=np.load(f_RID_Diag)
	RID=RID_vs_Diag[:,0]
	Diag=RID_vs_Diag[:,1]

	


	### load the rules
	rule_list_data=pd.read_csv(filename_rules, header=None)
	rule_list_mat=rule_list_data.as_matrix()
	rule_list = rule_list_mat.ravel().tolist()
	

	# Generate itemset corresponding to each RID and then file write and load it	
	generate_Itemset(X, filename_IS)
	data_df=pd.read_csv(filename_IS, sep=' ', header=None)
	data=data_df.as_matrix() # 'data' contains itemsets corresponding to each 'RID'
	N=data.shape[0] # Number of RIDs


	file_csv=open(f_Actual_Rules_Per_RID, 'w')
	file_wr=csv.writer(file_csv)
	header_string=['RID', 'diag_act', 'diag_pred', 'score', 'probability', 'itemsets',  'rules']
	file_wr.writerow(header_string) # Write the file header

	# Dict_RID_FS=collections.OrderedDict() # Dictionary to contain all the rules associated with each 'RID'
	for i in range(N):
		tmp=data[i][pd.notnull(data[i])].astype(np.int32)		
		diag_act=diag_dict2[Diag[i]]
		diag_pred=diag_dict3[label_pred_all[i]]
		score=result_train[i,0]
		probability=pred_proba_cal[i]
		
		
		itemsets=[]
		rules_all=[]		
			
		for item in tmp:						
			itemsets.append(item)		
			rule=rule_list[int(item)]	
			rules_all.append([rule])
	

	
		buf=[RID[i], diag_act, diag_pred, score, probability, itemsets, rules_all]			
		file_wr.writerow(buf)

	file_csv.close()















