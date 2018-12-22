from __future__ import division
import numpy as np
import subprocess
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import shutil


def get_Classification_Results(label_pred_all,y):
	correctly_classified=0
	misclassified=0
	pos_correctly_classified=0
	neg_correctly_classified=0
	pos_misclassified=0
	neg_misclassified=0
	# indx_misclassified=[]
	
	for i in range(y.shape[0]):	
		if(label_pred_all[i]==y[i]):
			correctly_classified+=1
			if(y[i]==+1):
				pos_correctly_classified+=1
			else:
				neg_correctly_classified+=1			
		else:
			misclassified+=1
			# indx_misclassified.append([i])
			if(y[i]==+1):
				pos_misclassified+=1
			else:
				neg_misclassified+=1
	# print indx_misclassified
	return [correctly_classified, misclassified, pos_correctly_classified, neg_correctly_classified, pos_misclassified, neg_misclassified ]



def get_score(path_data, path_labels, opt_rules):
	# read the data file
	data_all = []
	with open(path_data) as file:
		for line in file:
			tmp = line.split()
			data_all.append(tmp)


	data_all_arr = np.array(data_all)
	colnames_all = data_all_arr[:,0]
	data_val = data_all_arr[:, 1:].astype(int)
	# bit_vec = np.zeros(len(opt_rules), dtype=np.int8)
	bit_indx = [colnames_all.tolist().index(item) for item in opt_rules]
	bit_vec = data_val[bit_indx,:]
	pred_labels = np.any(bit_vec, axis=0)*1  
	


	label_all = []
	with open(path_labels) as file:
		for line in file:
			tmp = line.split()
			label_all.append(tmp)

	label_all_arr = np.array(label_all)
	true_labels = label_all_arr[1,:]
	true_labels = true_labels[1:].astype(int)
	# accuracy = accuracy_score(true_labels, pred_labels)
	[correctly_classified, misclassified, TP, TN, FN, FP ] = get_Classification_Results(pred_labels, true_labels)

	if((TP+FN)):
		SN=round(float(TP)/(TP+FN),2)
	
	if((TN+FP)):
		SP=round(float(TN)/(TN+FP),2)

	accuracy = 1 - round(float(misclassified)/true_labels.shape[0], 2)
	return [SN, SP, accuracy]







### Run CORELS
data_path="../data"
reg_par = 0.001
reg_par_val = str(reg_par) + '0000'


path_tr = data_path + '/compas_train.out'
path_tr_label = data_path + '/compas_train.label'

path_te = data_path + '/compas_test.out'
path_te_label = data_path + '/compas_test.label'		


# load optimum rules
logfile = '../logs/for-compas_train.out-curious_lb-with_prefix_perm_map-no_minor-removed=none-max_num_nodes=100000-c='+ reg_par_val +'-v=0-f=1000-opt.txt'
# delete the old logfile first
if os.path.exists('../logs'):
	shutil.rmtree('../logs')

os.makedirs('../logs')
cmd = "./corels -r " + str(reg_par) + " -c 2 -p 1 " + path_tr + " " + path_tr_label
subprocess.check_output([cmd], shell=True)


rule_data = pd.read_csv(logfile, delimiter=';', dtype=str, header=None)
# rule_data = pd.read_csv('../logs/for-ADNI_train.out-curious_obj-with_prefix_perm_map-no_minor-removed=none-max_num_nodes=100000-c=0.0050000-v=0-f=1000-opt.txt', delimiter=';', dtype=str, header=None)
rule_list = rule_data.as_matrix().ravel().tolist()

default_rule = rule_list[-1]
indx=default_rule.find('~') + 1
default_label = int(default_rule[indx:indx+1])

opt_rules = [item[:item.find('~')] for item in rule_list[:-1] ] # ignore the default rule
rule_len = len(opt_rules)

[SN_tr, SP_tr, accuracy_tr] = get_score(path_tr, path_tr_label, opt_rules)
[SN_te, SP_te, accuracy_te] = get_score(path_te, path_te_label, opt_rules)

print('SN_tr = {}, SP_tr= {}, accuracy_tr = {}'.format(SN_tr, SP_tr, accuracy_tr))
print('SN_te = {}, SP_te= {}, accuracy_te = {}'.format(SN_te, SP_te, accuracy_te))

		





