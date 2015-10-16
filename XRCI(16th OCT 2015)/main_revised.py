import string
import sys
import numpy as np
import pandas as pd
import csv
#Taking command line arguments
arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]


class read():
	def __init__(self):
 		#reading csv files
		self.W=pd.read_csv(arg1)
		self.X=pd.read_csv(arg2)
		self.Y=pd.read_csv(arg3)
		self.Z=pd.read_csv(arg4)
		#For testing the code, you can comment the above 4 lines and uncomment the 5 lines below 
		
		#Right now I have taken a wine.csv file and copied and renamed it to the 4 csv file below 
		#data is imported and count function is called which counts non NaN values,
		#in the clean class pandas interpolate function is used and then count is called again to show that NaN values have been replaced
		#with interpolated values
		'''self.W=pd.read_csv('id_time_vitals_train.csv')
		self.X=pd.read_csv('id_time_labs_train.csv')
		self.Y=pd.read_csv('id_age_train.csv')
		self.Z=pd.read_csv('id_label_train.csv')
		print(self.Z)
		print(self.Z.count())'''

class clean():
	def __init__(self):
		#reading dataset
		var=read()		
		n1=var.W[((var.W)-(var.W).mean()).abs()<=(3*(var.W).std())]
		n2=var.X[((var.X)-(var.X).mean()).abs()<=(3*(var.X).std())]
		n3=var.Y[((var.Y)-(var.Y).mean()).abs()<=(3*(var.Y).std())]
		n4=var.Z[((var.Z)-(var.Z).mean()).abs()<=(3*(var.Z).std())]
		#keep only the ones that are within +3 to -3 standard deviations in the column 'Data' and remaining to be replaced by NaN.
		#Un-comment the following line to test                
		#print(n1)
                self.W1=n1.interpolate()
		self.X1=n2.interpolate()
		self.Y1=n3.interpolate()
		self.Z1=n4.interpolate()
		#one drawback of interpolate:the NaN values before the first data point are ignored and remains NaN.So Sort it in descending order to make sure all the NaN values comes at the end of all the data
		#Un-comment the following line to test		
		#print(self.W1)
		n1 = self.W1.sort(columns=["TIME"], ascending=False, axis=0)
		n2 = self.X1.sort(columns=["ID"], ascending=False, axis=0)		
		n3 = self.Y1.sort(columns=["ID"], ascending=False, axis=0)		
		n4 = self.Z1.sort(columns=["ID"], ascending=False, axis=0)
		#Un-comment the following line to test		
		#print(n1)
		#interpolate it again to remove the NaN values at the ending of table	
		self.W1=n1.interpolate()
		self.X1=n2.interpolate()
		self.Y1=n3.interpolate()
		self.Z1=n4.interpolate()
		#sort according to your requirement
		n1 = self.W1.sort(columns=["TIME"], ascending=True, axis=0)
		n2 = self.X1.sort(columns=["ID"], ascending=True, axis=0)		
		n3 = self.Y1.sort(columns=["ID"], ascending=True, axis=0)		
		n4 = self.Z1.sort(columns=["ID"], ascending=True, axis=0)
		#Un-comment the following line to test 
		#print(n1)
		#function call
		
class training():
	def __init__(self):
		var1=clean()


t=training()

