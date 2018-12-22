#####################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
import csv
import collections



input_data_path="../Data/"



def create_Feature_Dictionary(path_dict):
	propublica_dict=collections.OrderedDict()

	## List of 14 features ####
	propublica_dict['Male']= 'sex:Male'
	propublica_dict['age:18-20']= 'age:18-20'
	propublica_dict['age:21-22']='age:21-22'
	propublica_dict['age:23-25']='age:23-25'
	propublica_dict['age:26-45']='age:26-45'
	propublica_dict['age:>45']= 'age:>45'
	propublica_dict['JF:>0']= 'juvenile-felonies:>0'
	propublica_dict['JM:>0']= 'juvenile-misdemeanors:>0' 
	propublica_dict['JC:>0']='juvenile-crimes:>0 '
	propublica_dict['priors:2-3']= 'priors:2-3'
	propublica_dict['priors:0']= 'priors:0'
	propublica_dict['priors:=1']= 'priors:=1'
	propublica_dict['priors:>3']='priors:>3'
	propublica_dict['ccd:M']='current-charge-degree:Misdemeanor'

	np.save(path_dict,propublica_dict)



def read_Data(path_data, path_labels):
	# read the data file
	data_all = []
	with open(path_data) as file:
		for line in file:
			tmp = line.split()
			data_all.append(tmp)

	data_all_arr = np.array(data_all)
	colnames_all = data_all_arr[:,0]
	data_val = data_all_arr[:, 1:].astype(int)
	
	# read the class labels file
	label_all = []
	with open(path_labels) as file:
		for line in file:
			tmp = line.split()
			label_all.append(tmp)

	label_all_arr = np.array(label_all)
	true_labels = label_all_arr[1,:]
	true_labels = true_labels[1:].astype(int)

	return [data_val, true_labels, colnames_all]


def load_data(data_path):
	# Create a feature dictionary required for visualization
	path_dict=data_path+'propublica_dict.npy'
	create_Feature_Dictionary(path_dict)
	Feature_dict=np.load(path_dict).item()	

	# Define Class labels dictionary
	class_labels_dict={-1:'recidivate-within-two-years:No', +1:'recidivate-within-two-years:Yes', 0:'Rejected'}
	

	### Read the train data
	[X, y, colnames_tr] = read_Data('../Data/compas_train.out', '../Data/compas_train.label')
	[X_test, y_test, colnames_te] = read_Data('../Data/compas_test.out', '../Data/compas_test.label')
	

	y = label_binarize(y, classes=[0,1], neg_label=-1, pos_label=1).ravel()
	y_test = label_binarize(y_test, classes=[0,1], neg_label=-1, pos_label=1).ravel()	

	RID_tr = np.arange(X.shape[1])
	RID_te = np.arange(X_test.shape[1])



	data_train=np.hstack((RID_tr[:,np.newaxis],y[:,np.newaxis],X.T))
	data_test=np.hstack((RID_te[:,np.newaxis],y_test[:,np.newaxis],X_test.T))	
	np.save('../Data/data_propublica.npy',[data_train, data_test, Feature_dict, class_labels_dict])

	### Write rules into file
	filename_rules= data_path + 'Rule_List_ProPublica.csv'
	csv_rule_list=open(filename_rules,'w')
	rule_writer=csv.writer(csv_rule_list)
	
	for i in colnames_tr:
		rule_writer.writerow([i])
	csv_rule_list.close()




	return [data_train, data_test, Feature_dict, class_labels_dict]


load_data(input_data_path)








