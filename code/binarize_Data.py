### Binarize data using non-overlapping bins ###
import numpy as np
import collections
import csv


def binarize(X, X_test, n_bins, Feature_nbins_dict, data_path):		
	X_all=np.vstack((X, X_test))

	cols_used=[]

	for key in Feature_nbins_dict.keys():
		cols_used.append(key)

	
	# cols_used=Feature_nbins_dict.keys()	

	### Create 'feature_range_array'  ####
	col_mins=np.min(X_all,axis=0)[:, np.newaxis]
	col_maxs=np.max(X_all,axis=0)[:, np.newaxis]
	feature_names=np.array(cols_used)[:, np.newaxis]
	feature_range_array=np.hstack((feature_names, col_mins))
	feature_range_array=np.hstack((feature_range_array, col_maxs))	
	np.save(data_path + 'Feature_range_array.npy', feature_range_array)

	
	### Binarize the plasma data
	rule_list=[]
	X_bins_th_all=[]	
	count_empty_list=0
	for j in range(X.shape[1]):		
		
		X_bins_th=np.linspace(np.min(X_all[:,j]),np.max(X_all[:,j]),n_bins)
		X_bins_th_all.append(X_bins_th)
		for i in range(n_bins-1):
			# print 'i: '+str(i)
			
			start=X_bins_th[i]
			end=X_bins_th[i+1]

			if(i==n_bins-2):				
				rule_list.append(str('%.2f' %start) + '<=' + cols_used[j] +'<=' + str('%.2f' %end))
				index_bin_train= np.where( np.logical_and(X[:,j]>=start, X[:,j]<=end) )[0]
				index_bin_test= np.where( np.logical_and(X_test[:,j]>=start, X_test[:,j]<=end) )[0]
			else:				
				rule_list.append(str('%.2f' %start) + '<=' + cols_used[j] +'<' + str('%.2f' %end))
				index_bin_train= np.where( np.logical_and(X[:,j]>=start, X[:,j]<end) )[0]
				index_bin_test= np.where( np.logical_and(X_test[:,j]>=start, X_test[:,j]<end) )[0]

			array_buf=np.array([ cols_used[j], start, end ])[np.newaxis, :]

			
			buf_train=np.zeros(X.shape[0]).reshape(X.shape[0],1) # a column buffer to hold the binary data (train)
			if(len(index_bin_train)>0):
				buf_train[index_bin_train]=1

			buf_test=np.zeros(X_test.shape[0]).reshape(X_test.shape[0],1) # a column buffer to hold the binary data (test)
			if(len(index_bin_test)>0):
				buf_test[index_bin_test]=1


			if(j==0 and i==0):
				X_binary=buf_train
				X_test_binary=buf_test
				rule_array=array_buf
			else:
				X_binary=np.hstack((X_binary,buf_train))
				X_test_binary=np.hstack((X_test_binary,buf_test))
				rule_array=np.vstack((rule_array, array_buf))	
	
	



	np.save(data_path + 'Rule_array.npy', rule_array)
		

	### Write rules into file
	filename_rules= data_path + 'Rule_List_' + 'nbins_'+ str(n_bins) + '.csv'
	csv_rule_list=open(filename_rules,'w')
	rule_writer=csv.writer(csv_rule_list)
	for i in rule_list:
		rule_writer.writerow([i])
	csv_rule_list.close()


	np.save(data_path+'X_all_binary_train.npy',X_binary)
	np.save(data_path+'X_all_binary_test.npy',X_test_binary)

	return [X_binary, X_test_binary]










