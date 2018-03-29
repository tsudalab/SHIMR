from __future__ import division
import numpy as np
import csv
import os


class Optimizer_Primal(object):
	def __init__(self, Y, d):		
		self.Y=Y
		self.d=d
		
	
	def generate_opt_fun(self, filename, X, C_POS, C_NEG):
		self.X=X
		N=self.X.shape[0]
		M=self.X.shape[1]		
		d=self.d
		Y=self.Y		
		
		buf_bound1=[]
		buf_bound2=[]
		
		obj1_sum=''
		for j in range(M):
			if(obj1_sum!=''):
				sign='+'
			else:
				sign=''			
			obj1_sum+= sign + ' '  + 'A'+str(j) + ' + '  + 'B'+str(j) +  ' ' # 1st term of the Objective
			
			buf_bound1.append(' ' +  'A'+str(j) +   ' >= ' +  str(0)  +  '\n')	# Bound 1
			buf_bound2.append(' ' +  'B'+str(j) +   ' >= ' +  str(0)  +  '\n') # Bound 2			


		const_indx=1;
		buf_const1=[]
		buf_const2=[]
		buf_bound3=[]
		buf_bound4=[]		
		Xi_sum_obj=''
		gamma_sum_obj=''
		
		for i in range(N):  # For each example	
			if(Y[i]==+1):
				C=C_POS
			else:
				C=C_NEG

			if(Xi_sum_obj!=''):
				sign='+'
			else:
				sign=''

			Xi_sum_obj+= sign + ' ' + '%0.5f' %C + ' '  + 'Xi_' + str(i) + ' ' # 2nd term of the Objective
			gamma_sum_obj+= sign + ' ' + '%0.5f'%(((1-2*self.d)/(self.d))*C) + ' ' + 'gamma_' +str(i) + ' ' # 3rd term of the Objective
			
		
			buf_bound3.append(' ' +  'Xi_'+str(i) +   ' >= ' +  str(0)  +  '\n') # Bound 3
			buf_bound4.append(' ' +  'gamma_'+str(i) +   ' >= ' +  str(0)  +  '\n') # Bound 4
	
			const1_sum=''				
			for j in range(M):			
				val1=Y[i]*self.X[i,j]	# First term (A)			
				
				if(val1<0):
					val1=np.fabs(val1)
					sign='-'
				else:
					val1=np.fabs(val1)
					sign='+'					

				const1_sum+= sign + ' ' + str(val1)  +  ' '  + 'A'+str(j) +  ' '

				val2=(-1)*Y[i]*self.X[i,j]  # Second term (B)
				if(val2<0):
					val2=np.fabs(val2)
					sign='-'
				else:
					val2=np.fabs(val2)
					sign='+'

				const1_sum+= sign + ' ' + str(val2)  +  ' '  + 'B'+str(j) +  ' '		

			
			val3=Y[i]
			if(val3<0):				
				sign='-'
			else:				
				sign='+'

				
			### 1st constraint
			buf_const1.append(' c'+ str(const_indx) + ': '  + const1_sum   +  sign +  ' b ' + '+ ' + 'Xi_'+str(i)+' >= ' + ' ' + str(1) +'\n')
			const_indx+=1

			### 2nd constraint
			buf_const2.append(' c'+ str(const_indx) + ': '  + const1_sum   +  sign +  ' b ' + '+ ' + 'gamma_'+str(i)+' >= ' + ' ' + str(0) +'\n')
			const_indx+=1			
		
		




		buf=[]

		# Write the Objective
		buf.append('Minimize'+'\n')
		obj2_sum = Xi_sum_obj + '+' + gamma_sum_obj
		buf.append(' obj: ' + obj1_sum   + ' + ' +   obj2_sum  + '\n')

		# Write the Constraints
		buf.append('Subject To'+'\n')
		buf.extend(buf_const1)
		buf.extend(buf_const2)

		### Write the Bounds	
		buf.append('Bounds'+'\n')	
		buf.extend(buf_bound1)
		buf.extend(buf_bound2)
		buf.extend(buf_bound3)
		buf.extend(buf_bound4)

		buf.append('End')


		# Open file writer to create the file
		file_opt=open(filename, 'w')		
		file_opt.write("".join(buf))
		file_opt.close()






