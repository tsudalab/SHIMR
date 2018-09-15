#####################################################
from __future__ import division
import numpy as np
import csv
import os




class Optimizer(object):
	def __init__(self, Y, d):		
		self.Y=Y
		self.d=d
		
	
	def generate_opt_fun(self, filename, X, C_POS, C_NEG):
		self.X=X
		N=self.X.shape[0]
		M=self.X.shape[1]		
		
		d=self.d
		Y=self.Y	

		
		P_sum_obj='' 
		buf_bound=[] 
		const3_sum=''
		for i in range(N):
			# 1st term of the Objective
			if(P_sum_obj!=''):
				sign='+'
			else:
				sign=''
			P_sum_obj+= sign + ' '  + 'P' + str(i) + ' '


			# Constraint 3
			val=Y[i] 
			if(val<0):
				val=np.fabs(val)
				sign='-'
			else:
				val=np.fabs(val)
				sign='+'
			const3_sum+= sign + ' ' + str(val)  +  ' '  + 'P'+str(i) +  ' '
			const3_sum+= sign + ' ' + str(val)  +  ' '  + 'Q'+str(i) +  ' '

			# Bound 1 and Bound 2
			if(Y[i]==+1):
				C=C_POS
			else:
				C=C_NEG
		
			buf_bound.append(' ' + str(0) + '<=' + 'P'+str(i) +   '<=' +  str(C)  +  '\n') # Bound 1
			buf_bound.append(' ' + str(0) + '<=' + 'Q'+str(i) +   '<=' +  str(C*(1-2*d)/d)  +  '\n') # Bound 2


		


		const_indx=1;		
		buf_const1=[]
		buf_const2=[]
		for j in range(M):  # For each column				
			const1_sum=''
			const2_sum=''
			for i in range(N):
				# 1st constraint (for +Ve part of the modulus)
				val=Y[i]*self.X[i,j] 				
				if(val<0):
					val=np.fabs(val)
					sign='-'
				else:
					val=np.fabs(val)
					sign='+'	
				const1_sum+= sign + ' ' + str(val)  +  ' '  + 'P'+str(i) +  ' '
				const1_sum+= sign + ' ' + str(val)  +  ' '  + 'Q'+str(i) +  ' '

				# 2nd constraint (for -Ve part of the modulus)
				val=(-1)*Y[i]*self.X[i,j] 				
				if(val<0):
					val=np.fabs(val)
					sign='-'
				else:
					val=np.fabs(val)
					sign='+'	
				const2_sum+= sign + ' ' + str(val)  +  ' '  + 'P'+str(i) +  ' '
				const2_sum+= sign + ' ' + str(val)  +  ' '  + 'Q'+str(i) +  ' '


			buf_const1.append(' c'+str(const_indx) + ': ' + const1_sum  +  '<=' + ' ' + str(1)   +  '\n')
			const_indx+=1	

			buf_const2.append(' c'+str(const_indx) + ': ' + const2_sum  +  '<=' + ' ' + str(1)   +  '\n')
			const_indx+=1

		
		buf=[]
		buf.append('Maximize'+'\n') # Write the Objective
		buf.append(' obj: ' + P_sum_obj   + '\n')

		buf.append('Subject To'+'\n')   # Write the constraints
		buf.extend(buf_const1)
		buf.extend(buf_const2)		
		buf.append(' c'+str(const_indx) + ': ' + const3_sum  +  '=' + ' ' + str(0)   +  '\n')
				
		buf.append('Bounds'+'\n') # Write the Bounds
		buf.extend(buf_bound)			
		buf.append('End')

		# Open file writer to create the file
		file_opt=open(filename, 'w')
		file_opt.write("".join(buf))  # Now file write everything at once
		file_opt.close()





















