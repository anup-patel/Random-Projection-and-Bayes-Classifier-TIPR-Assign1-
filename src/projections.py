import sys
from nn import *
from bayes import *
import numpy as np
import csv
import random
import sklearn
import math
import operator
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
#import matplotlib.pyplot as plt  
import timeit
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
#Implementing Random Projection Algorithm

def projections(data,k):
	d=len(data[0])
	#Implementing Random Projection Algorithm
	#k=int(input("Enter the new Dimension :: "))
	#Generating Random Matrix with Mean 0 and Variance 1
	random_matrix= 1 * np.random.randn(d, k) + 0
	#print(len(random_matrix))
	#print(len(random_matrix[0]))
	#print(random_matrix)

	projection=(np.matmul(data,random_matrix))/np.sqrt(k)
	#print("Projection Matrix is")
	#print(len(projection))

	#print(projection[1])
	'''for i in range(1,10):
		accuracy=classify_custom(projection,label,k)
		print("Accuracy for K= "+str(i)+" is: "+str(accuracy))
		f.write("Accuracy for K= "+str(i)+" is: "+str(accuracy))
		f.write("\n")'''

	#print("Using Custom KNN")
	#accuracy=classify_custom(projection,label,3)
	return projection
