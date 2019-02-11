import sys
from projections import *
from bayes import *
import numpy as np
import csv
import random
import sklearn
import math
import operator
#import matplotlib.pyplot as plt  
import timeit
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

#KNN Implementation Using SK- Learn
def classify(data,label,k):
    start = timeit.default_timer()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.30) 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) 
    #print(X_train) 
    X_test = scaler.transform(X_test) 
    #print(y_train)
    classifier = KNeighborsClassifier(n_neighbors=k)  
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test) 
    accuracy_sk=sklearn.metrics.accuracy_score(y_test,y_pred)*100.0
    #print("### Sklearn KNN Implementation")
    print("Test Accuracy :: "+str(round(accuracy_sk,2)))
    print("Macro F1 Score is :: "+ str(round(sklearn.metrics.f1_score(y_test, y_pred, average="macro")*100,2)))
    print("Micro F1 Score is :: "+ str(round(sklearn.metrics.f1_score(y_test, y_pred, average="micro")*100,2)))
    stop = timeit.default_timer()
    #print('Time: ', stop - start) 
    return accuracy_sk
    


#KNN Implementation Without SK Learn (Custom Implementation)

from scipy.spatial import distance

def eu_distance(test,train):
	test=np.delete(test,0) #Remove First Element of test array as it is label
	train=np.delete(train, 0) #Remove First Element of train array as it is label
	dist=distance.euclidean(test,train)
	return dist

def  getNeighbors(trainset,testins,k):
    start1 = timeit.default_timer()
    distances=[]
    for x in range(len(trainset)):
        dist=eu_distance(testins,trainset[x])
        #print(dist)
        #trainset=np.column_stack([q[x], trainset[x]])
        #trainset[x]=np.insert(trainset[x],0,q[x])
        #testins=np.insert(testins,0,p)
        #print(testins)
        #testins=np.column_stack([p,testins])
        distances.append((trainset[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    #print("hi"+str(neighbors))
    stop1 = timeit.default_timer()
    #print('Time: ', stop1 - start1) 
    return neighbors
def getresponse(neighbors):
    voting={} #To check which has got maximum occurances
    for x in range(len(neighbors)):
        response=neighbors[x][0]
        if response in voting:
            voting[response]=voting[response]+1
        else:
            voting[response]=1
    sortedvotes=sorted(voting.items(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]

def getaccuracy(testset,prediction):
    correct=0
    for x in range(len(testset)):
        if (testset[x][0]==prediction[x]):
            correct+=1
    return (correct/float(len(testset)))*100.0

def classify_custom(data,label,k):
    start = timeit.default_timer()
    split=0.7
    rand=False
    trainingset=[]
    testset=[]
    p=[]
    t=[]
    data=np.column_stack([label, data])
    #print(len(data[0]))
    if(rand):
        for x in range(len(data)):
            if(random.random()<split):
                trainingset.append(data[x])
            else:
                testset.append(data[x])
    else:
        train=int(len(data)*split)
        for i in range(train):
            trainingset.append(data[i])
        for i in range(train,len(data)):
            testset.append(data[i])
         
    
    # generate predictions
    predictions=[]
   
    for x in range(len(testset)):
        neighbors = getNeighbors(trainingset, testset[x], k)
        result = getresponse(neighbors)
        predictions.append(result)
        #print(str(x)+'> predicted=' + str(result) + ', actual=' + str(testset[x][0]))
    
    #accuracy=sklearn.metrics.accuracy_score(testset, predictions)*100.0
    accuracy = getaccuracy(testset, predictions)
    for x in range(len(testset)):
        t.append(testset[x][0])
        p.append(predictions[x])

    #stop = timeit.default_timer()
    #print("### Custom KNN Implementation")
    print("Test Accuracy :: "+str(round(accuracy,2)))
    print("Test Macro F1-score :: "+ str(round((sklearn.metrics.f1_score(t, p, average="macro")*100),2)))
    print("Test Micro F1-score :: "+ str(round((sklearn.metrics.f1_score(t, p, average="micro")*100),2)))
    #print('Time: ', stop - start) 
    return accuracy

def cross_validation(data,label,fold):
	from sklearn.model_selection import KFold
	
	#"data" store my data sample
	# prepare cross validation
	data=np.column_stack([label, data])
	kfold = KFold(fold, True, 1) #Value 1 for pseudorandom Generator
	# enumerate splits
	rn=1
	
	for train, test in kfold.split(data):
		testset=data[test]
		trainingset=data[train]
		y_train=[] #Label of Training Set
		y_test=[] #Label of Test Set
		#print('train: %s, test: %s' % (data[train], data[test]))
		# generate predictions
		for i in range(len(trainingset)):
			y_train.append(trainingset[i][0])
		for j in range(len(testset)):
			y_test.append(testset[j][0])
		
		X_train=np.delete(trainingset,0,axis=1)
		X_test=np.delete(testset,0,axis=1)
		scaler = StandardScaler()  
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)  
		#print(len(X_train))
		#print((y_train))
		X_test = scaler.transform(X_test) 
		classifier = KNeighborsClassifier(n_neighbors=10)  
		classifier.fit(X_train,y_train) 
		y_pred = classifier.predict(X_test) 
		accuracy_sk=sklearn.metrics.accuracy_score(y_test,y_pred)*100.0
		print("Round :: "+str(rn))
		rn=rn+1
		print("Test Accuracy :: "+str(round(accuracy_sk,2)))
		print("Macro F1 Score is :: "+ str(round(sklearn.metrics.f1_score(y_test, y_pred, average="macro")*100,2)))
		print("Micro F1 Score is :: "+ str(round(sklearn.metrics.f1_score(y_test, y_pred, average="micro")*100,2)))
		stop = timeit.default_timer()
		#print('Time: ', stop - start) 
		#return accuracy_sk


		
