import sys
from nn import *
from projections import *
from bayes import *
#from twitter import *
import numpy as np
import csv
import random
import sklearn
import math
import operator
import re
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
#import matplotlib.pyplot as plt  
import timeit
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from sklearn import datasets
if __name__ == '__main__':



	def twitter():
		inputfile1=open("../data/twitter/twitter.txt","r")
		inputlines1=inputfile1.readlines()
		inputfile1.close() 
		fileName = "../data/twitter/twitter_label.txt"
		inputfile2=open(fileName,"r")
		inputlines2=inputfile2.readlines()
		inputfile2.close()



		B=[0]*len(inputlines2)
		for i in range(len(B)):
			B[i]=list(map(float,inputlines2[i].split(" ")))
		 
		label=[]
		for i in range(0,len(B)):
		    label.append(B[i][0])
		
		from sklearn.feature_extraction.text import TfidfVectorizer
		# list of text documents
		text = inputlines1
		# create the transform
		vectorizer = TfidfVectorizer()
		# tokenize and build vocab
		vectorizer.fit(text)
		# summarize
		#print(vectorizer.vocabulary_)
		#print(vectorizer.idf_)
		# encode document
		vector = vectorizer.transform([text[0]])
		# summarize encoded vector
		data=vector.toarray()
		#print(vector.shape)
		#print(vector.toarray())


		#if(fileName=="twitter/twitter_label.txt"):
		#	print("Anup")
		#Functions for Twitter Data Processing
		def tokenize(sentences):
		    words = []
		    for sentence in sentences:
		        w = word_extraction(sentence)
		        words.extend(w)
		        
		    words = sorted(list(set(words)))
		    return words

		def word_extraction(sentence):
		    ignore = ['a', "the", "is"]
		    words = re.sub("[^\w]", " ",  sentence).split()
		    cleaned_text = [w.lower() for w in words if w not in ignore]
		    return cleaned_text      

		def generate_bow(allsentences):    
			vocab = tokenize(allsentences)
			
			data=[[0] * len(vocab) for t in range(len(allsentences))]
			p=0
			#print("Word List for Document \n{0} \n".format(vocab))
			for sentence in allsentences:
				words = word_extraction(sentence)
				bag_vector = np.zeros(len(vocab))
				for w in words:
					for i,word in enumerate(vocab):
						if (word == w): 
							bag_vector[i] += 1
							#print(bag_vector[i])
					
						#print(bag_vector[i])
				#print(sentence)
				#print(np.array(bag_vector))
				data[p]=bag_vector
				p=p+1
			return data

		allsentences = inputlines1
		data=generate_bow(allsentences)
		print("### Bag of Word Problem")
		bayes_sklearn(data,label)

	def dolphins():
		#For Dataset other than Twitter
		inputfile1=open("../data/dolphins/dolphins.csv","r")
		inputlines1=inputfile1.readlines()
		inputfile1.close() 
		fileName = "../data/dolphins/dolphins_label.csv"
		inputfile2=open(fileName,"r")
		inputlines2=inputfile2.readlines()
		inputfile2.close()
		
		A=[0]*len(inputlines1)
		for i in range(len(A)):
			A[i]=list(map(float,inputlines1[i].split(" ")))

		B=[0]*len(inputlines2)
		for i in range(len(B)):
			B[i]=list(map(float,inputlines2[i].split(" ")))
		 
		label=[]
		for i in range(0,len(B)):
		    label.append(B[i][0])


	#For other Datasets    
		print('Welcome to the world of high and low dimensions!')
		#print("Size of Input Matrix is:"+ str(len(A))+"*"+str(len(A[0])))
		#print(len(B))
		#Given Dimension of Matrix is 
		#d=len(A[0])
		#print("Dimension of Given Data is :: " + str(d))

		################################################################
	
		for pr in range(1,math.floor(len(A[0])/4)):
			pr=pr*2
			proj=projections(A,pr) #Projection Matrix
			print("### Projection Matrix Details for Dimension :: "+ str(pr))
			classify(proj,label,10) #Usually we take k=10 (No Reason)
			
			
		
		################################################################

		print("#Custom KNN Implementation")
		classify_custom(A,label,10) #KNN Custom Implementation

		###############################################################

		print("#Sklearn KNN Implementation")
		classify(A,label,10) #KNN Using Sklearn

		###############################################################
		
		print("#Custom Bayes Classifier Implementation")
		bayes(A,label) #Bayes Classifier Custom Implementation

		###############################################################

		print("#Sklearn Bayes Classifier Implementation")
		bayes_sklearn_G(A,label) #Bayes Classifier Using Sklearn

		###############################################################

		print("Cross Validation")
		cross_validation(A,label,3)
	 
		###############################################################


	def pubmed():
		#For Dataset other than Twitter
		inputfile1=open("../data/pubmed/pubmed.csv","r")
		inputlines1=inputfile1.readlines()
		inputfile1.close() 
		fileName = "../data/pubmed/pubmed_label.csv"
		inputfile2=open(fileName,"r")
		inputlines2=inputfile2.readlines()
		inputfile2.close()
		
		A=[0]*len(inputlines1)
		for i in range(len(A)):
			A[i]=list(map(float,inputlines1[i].split(" ")))

		B=[0]*len(inputlines2)
		for i in range(len(B)):
			B[i]=list(map(float,inputlines2[i].split(" ")))
		 
		label=[]
		for i in range(0,len(B)):
		    label.append(B[i][0])



		################################################################
	
		proj=projections(A) #Projection Matrix
		print("### Projection Matrix Details")
		classify(proj,label,10) #Usually we take k=10 (No Reason)
		
		################################################################

		#print("#Custom KNN Implementation")
		#classify_custom(A,label,10) #KNN Custom Implementation

		###############################################################

		print("#Sklearn KNN Implementation")
		classify(A,label,10) #KNN Using Sklearn

		###############################################################
		
		#print("#Custom Bayes Classifier Implementation")
		#bayes(A,label) #Bayes Classifier Custom Implementation

		###############################################################

		print("#Sklearn Bayes Classifier Implementation")
		bayes_sklearn(A,label) #Bayes Classifier Using Sklearn

		###############################################################

		print("#Cross Validation")
		cross_validation(A,label,3)
	 
		###############################################################


	#f= open(sys.argv[1],'r')
	if(sys.argv[1]=="twitter"):
		twitter()
	elif(sys.argv[1]=="pubmed"):
		pubmed()
	elif(sys.argv[1]=="dolphins"):
		dolphins()



	#lsh(A,label) #LSH Implementation

	
