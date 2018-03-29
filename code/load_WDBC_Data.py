import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
import csv
import collections



input_data_path="../Data/"



def create_Feature_Dictionary(path_dict):
	wdbc_dict=collections.OrderedDict()

	## List of 30 features ####
	wdbc_dict['Rad_M']= 'Radius Mean'
	wdbc_dict['Text_M']= 'Texture Mean'
	wdbc_dict['Peri_M']='Perimeter Mean'
	wdbc_dict['Area_M']='Area Mean'
	wdbc_dict['Smoo_M']='Smoothness Mean'
	wdbc_dict['Com_M']= 'Compactness Mean'
	wdbc_dict['Con_M']= 'Concavity Mean'
	wdbc_dict['ConP_M']= 'Concave Points Mean' 
	wdbc_dict['Sym_M']='Symmetry Mean'
	wdbc_dict['Frac_M']= 'Fractal Dimension Mean'


	wdbc_dict['Rad_SE']= 'Radius Standard Error'
	wdbc_dict['Text_SE']= 'Texture Standard Error'
	wdbc_dict['Peri_SE']='Perimeter Standard Error'
	wdbc_dict['Area_SE']='Area Standard Error'
	wdbc_dict['Smoo_SE']='Smoothness Standard Error'
	wdbc_dict['Com_SE']= 'Compactness Standard Error'
	wdbc_dict['Con_SE']= 'Concavity Standard Error'
	wdbc_dict['ConP_SE']= 'Concave Points Standard Error' 
	wdbc_dict['Sym_SE']='Symmetry Standard Error'
	wdbc_dict['Frac_SE']= 'Fractal Dimension Standard Error'

	wdbc_dict['Rad_W']= 'Radius Worst'
	wdbc_dict['Text_W']= 'Texture Worst'
	wdbc_dict['Peri_W']='Perimeter Worst'
	wdbc_dict['Area_W']='Area Worst'
	wdbc_dict['Smoo_W']='Smoothness Worst'
	wdbc_dict['Com_W']= 'Compactness Worst'
	wdbc_dict['Con_W']= 'Concavity Worst'
	wdbc_dict['ConP_W']= 'Concave Points Worst' 
	wdbc_dict['Sym_W']='Symmetry Worst'
	wdbc_dict['Frac_W']= 'Fractal Dimension Worst'


	
	np.save(path_dict,wdbc_dict)

# header=['RID','Diagnosis','Rad_M','Text_M','Peri_M','Area_M','Smoo_M','Com_M','Con_M','ConP_M','Sym_M','Frac_M',
# 	    'Rad_SE','Text_SE','Peri_SE','Area_SE','Smoo_SE','Com_SE','Con_SE','ConP_SE','Sym_SE','Frac_SE',
# 	    'Rad_W','Text_W','Peri_W','Area_W','Smoo_W','Com_W','Con_W','ConP_W','Sym_W','Frac_W']




def load_data(data_path):
	# Create a feature dictionary required for visualization
	path_dict=data_path+'wdbc_dict.npy'
	create_Feature_Dictionary(path_dict)
	Feature_dict=np.load(path_dict).item()	

	# Define Class labels dictionary
	class_labels_dict={-1:'Benign', +1:'Malignant', 0:'Rejected'}
	
	### Read the train data
	wdbc_train_df=pd.read_csv(input_data_path+'wdbc_train.csv', sep=',')
	wdbc_train=wdbc_train_df.as_matrix()

	### Read the test data
	wdbc_test_df=pd.read_csv(input_data_path+'wdbc_test.csv', sep=',')
	wdbc_test=wdbc_test_df.as_matrix()
	
	
	X=wdbc_train[:,2:]
	y=wdbc_train[:,1] ### Diagnosis 

	X_test=wdbc_test[:,2:]
	y_test=wdbc_test[:,1]  ### Diagnosis 
		

	
	y=label_binarize(y, classes=['B','M'], neg_label=-1, pos_label=1).ravel()	
	y_test=label_binarize(y_test, classes=['B', 'M'], neg_label=-1, pos_label=1).ravel()




	data_train=np.hstack((wdbc_train[:,0][:,np.newaxis],y[:,np.newaxis],X))
	data_test=np.hstack((wdbc_test[:,0][:,np.newaxis],y_test[:,np.newaxis],X_test))	
	np.save('../Data/data.npy',[data_train, data_test, Feature_dict, class_labels_dict])



	return [data_train, data_test, Feature_dict, class_labels_dict]











