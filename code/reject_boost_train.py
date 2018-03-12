#### This is for only upto 3rd level of tree search in LCM used for multivariate analysis of iBoost with L1 penalty, size_L=1, size_U=3"


from __future__ import division
import numpy as np
from cplex_opt import *
import subprocess
import os
import pandas as pd
import time
from decimal import *






class Reject_Boost_Trainer(object):
    def __init__(self, data_path,size_U):         
        self.data_path=data_path
        self.size_L=1
        self.size_U=size_U

        

    def train(self, X, y, d, C_POS, C_NEG, Opt, filename_dual):
        MAX_ITERATION_COUNT=500
        itr_count=0        
        self.feature_set=[]         
        self.filepath_dual=self.data_path+filename_dual        
        N=X.shape[0]        
        

        filepath_wt_pos=self.data_path+'Reject_Boost_Weight_pos_supp.dat'
        filepath_wt_neg=self.data_path+'Reject_Boost_Weight_neg_supp.dat'
        filepath_lcm_out=self.data_path+'out.dat'  ### LCM output file
        filepath_tmp=self.data_path+"itemset_train.txt"     


        indx_transaction=self.generate_Itemset(X, filepath_tmp) # transactions which contain at least one itemset.
        self.indx_transaction=indx_transaction

        size_L=self.size_L
        size_U=self.size_U


        

        gain_updated_all=[]

        while(itr_count<MAX_ITERATION_COUNT):
            if(itr_count==0):
                self.lagrange_multipliers=np.ones(X.shape[0])              

           
                wt_boost_pos_supp=np.multiply(self.lagrange_multipliers.reshape((N,1)),y.reshape((N,1)))[indx_transaction]                
                np.savetxt(filepath_wt_pos,wt_boost_pos_supp)

               
                wt_boost_neg_supp=np.multiply(self.lagrange_multipliers.reshape((N,1)),y.reshape((N,1)))[indx_transaction]*(-1)
                np.savetxt(filepath_wt_neg,wt_boost_neg_supp)
                

                min_supp_pos=subprocess.check_output(["lcm53/lcm C -l "+str(size_L)+ " -u "+str(size_U)+" -K 1 "+" -w " + filepath_wt_pos + " "+filepath_tmp+" 1"], shell=True).strip()
                min_supp_neg=subprocess.check_output(["lcm53/lcm C -l "+str(size_L)+ " -u "+str(size_U)+" -K 1 "+" -w " + filepath_wt_neg + " "+filepath_tmp+" 1"], shell=True).strip()
               
                min_supp_pos=float(min_supp_pos)
                min_supp_neg=float(min_supp_neg)

                min_supp=max(min_supp_pos,min_supp_neg)
                if(min_supp==min_supp_pos):
                    wt_boost_file=filepath_wt_pos                
                else:
                    wt_boost_file=filepath_wt_neg
                    
                D = Decimal(str(min_supp))
                ep=D.as_tuple().exponent
                sub=10**ep
                min_supp=str(float(min_supp) - sub)
              
                subprocess.call(["lcm53/lcm Cf -l "+ str(size_L) + " -u "+str(size_U)+ " -w "+ wt_boost_file+ "  "+filepath_tmp+" "+ min_supp+" "+filepath_lcm_out], shell=True)### Call 'LCM'
              
                X_column=self.generate_Column(X, filepath_lcm_out)  ### Generate data column based on lcm output                
                           
                self.feature_set.append(self.feature_set_lcm)
               
                X_new=X_column ### Update the data column
                X_new=X_new.reshape(X_new.shape[0],1) ### Make it 2D array

                
                self._solve_LP_Dual(Opt, X_new, C_POS, C_NEG )
                
                gain_updated = self._get_Gain(X_new, y) 
                gain_updated_all.append(gain_updated)
                
                itr_count+=1
                
            else: 
                
                
                wt_boost_pos_supp=np.multiply(self.lagrange_multipliers.reshape((N,1)),y.reshape((N,1)))[indx_transaction]                
                np.savetxt(filepath_wt_pos,wt_boost_pos_supp)
                
                
                wt_boost_neg_supp=np.multiply(self.lagrange_multipliers.reshape((N,1)),y.reshape((N,1)))[indx_transaction]*(-1)                
                np.savetxt(filepath_wt_neg,wt_boost_neg_supp)
                             
                min_supp_pos=subprocess.check_output(["lcm53/lcm C -l "+str(size_L)+ " -u "+str(size_U)+" -K 1 "+" -w "+filepath_wt_pos+" "+filepath_tmp+" 1"], shell=True).strip()
             
                min_supp_neg=subprocess.check_output(["lcm53/lcm C -l "+str(size_L)+ " -u "+str(size_U)+" -K 1 "+" -w "+filepath_wt_neg+" "+filepath_tmp+" 1"], shell=True).strip()
                

                min_supp_pos=float(min_supp_pos)
                min_supp_neg=float(min_supp_neg)

                min_supp=max(min_supp_pos,min_supp_neg)
                if(min_supp==min_supp_pos):
                    wt_boost_file=filepath_wt_pos                    
                else:
                    wt_boost_file=filepath_wt_neg
                   

                D = Decimal(str(min_supp))
                ep=D.as_tuple().exponent
                sub=10**ep
                min_supp=str(float(min_supp) - sub)
               

        
                subprocess.call(["lcm53/lcm Cf -l "+ str(size_L) + " -u "+str(size_U)+ " -w "+ wt_boost_file+ "  "+filepath_tmp+" "+ min_supp+" "+filepath_lcm_out], shell=True) ### Call 'LCM'
                

                X_column=self.generate_Column(X, filepath_lcm_out)  ### Generate data column based on lcm output
               
                max_gain=self._get_Gain(X_column, y)
        
                if(max_gain>1 and self.feature_set_lcm not in self.feature_set):
                    self.feature_set.append(self.feature_set_lcm)                
                    
                    X_column=X_column.reshape((X_column.shape[0],1))                 
                    X_new=np.hstack((X_new,X_column))  ### Update the data column

                    self._solve_LP_Dual(Opt, X_new, C_POS, C_NEG)   
                    
                    
                    
                    gain_updated = self._get_Gain(X_new, y) 

                    gain_updated_all.append(gain_updated) 
                    

                    itr_count+=1
                else:
                    # print "No more hypothesis found !"
                    break
                

        ### File write updated gain ###
        gain_updated_all_dir=self.data_path + 'gain_updated_all/'
        if not os.path.exists(gain_updated_all_dir):
            os.makedirs(gain_updated_all_dir)            
        np.save(gain_updated_all_dir + 'gain_updated_all_d_'+ str(d) + '.npy', gain_updated_all)
        
        return self    

    ### Given the bonary data, this script generates itemset file. This is will be as input to "lcm" 
    def generate_Itemset(self, X, filename):      
        file_fs=open(filename, "w")  
        indx_transaction=[]  
        for i in range(X.shape[0]):
            indx=np.where(X[i,:]==1)[0]
            if indx.any():
                indx_transaction.append(i)
                for item in indx:
                    file_fs.write('%d' %item)
                    file_fs.write(' ')
                file_fs.write('\n')     
        file_fs.close()
        return indx_transaction

    ### Given the feature set, it generates the data column.
    def generate_Column(self, X, filepath_lcm_out):         
        data=pd.read_csv(filepath_lcm_out, header=None).as_matrix()
        wt_all=[]
        
        for i in range(data.shape[0]):
            tmp=data[i,0].split(' ')
            tmp_last=tmp[-1]
            tmp_last=float(tmp[-1][tmp_last.index('(')+1:tmp_last.index(')')])
            wt_all.append(tmp_last)
            

        top_indx=np.argmax(wt_all)
        itemset_top=data[top_indx,0].split(' ')[:-1]
        indx_feature_set=[int(val) for val in itemset_top]
       

        self.feature_set_lcm=indx_feature_set
        buf=np.prod(X[:,indx_feature_set], axis=1).reshape((X.shape[0],1))
        buf=2*buf-1       
        return buf
                    
                

    def generate_Column_2(self, X, feature_set_selected):       
        i=0 ### Columns index
        for indx_feature_set in feature_set_selected:
            if len(indx_feature_set)>0:
                buf=np.prod(X[:,indx_feature_set], axis=1)                
                buf=buf.reshape((X.shape[0],1))
                buf=2*buf-1
                if(i==0):
                    buf_all=buf
                else:
                    buf_all=np.hstack((buf_all,buf))
                i+=1  
        
        return buf_all

    
    

    def _solve_LP_Primal(self, Opt, X, C_POS, C_NEG, filename_primal):
        N=X.shape[0]         
        filepath_primal=self.data_path+filename_primal
        Opt.generate_opt_fun(filepath_primal, X, C_POS, C_NEG)  ### Generate input file for 'cplex' solver
       
        c=lpex2(filepath_primal,"p")
        
        A=[]
        B=[]
        
        for i, x in enumerate(c.variables.get_names()):
            if(x.startswith('A')):
                A.append(c.solution.get_values(i))
            if(x.startswith('B')):
                B.append(c.solution.get_values(i))
            if(x.startswith('b')):
                b=c.solution.get_values(i)

        self._bias=b
        lagrange_multipliers=np.array(A) - np.array(B)
        self.primal_values=np.array(lagrange_multipliers)                
        

                    

    def _solve_LP_Dual(self, Opt, X, C_POS, C_NEG):        
        N=X.shape[0]         
        Opt.generate_opt_fun(self.filepath_dual, X, C_POS, C_NEG)  ### Generate input file for 'cplex' solver       
        
        c=lpex2(self.filepath_dual,"p")
       
        sols = np.array(c.solution.get_values())
        P=sols[:N]
        Q=sols[N:]
       
        lagrange_multipliers = P + Q
        self.lagrange_multipliers=lagrange_multipliers



    def _get_Gain(self, X, y):
        gain_pos=0.0
        gain_neg=0.0
        gain=0.0

        N=X.shape[0]
        M=X.shape[1]

        for j in range(M): ### Loop through all the columns 
            X_pos=X[:,j]*[+1] 
            X_neg=X[:,j]*[-1] 
            for i in range(N):
                gain_pos+=X_pos[i]*self.lagrange_multipliers[i]*y[i]
                gain_neg+=X_neg[i]*self.lagrange_multipliers[i]*y[i]
            gain+=max(gain_pos,gain_neg)
        

        return gain
            




    def _predict(self, X, primal_values):
        """
        Computes the rejection boost prediction on the given features x.
        """      
      
        return self._bias + np.inner(X,primal_values)



    


   



    
       























