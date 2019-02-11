import sys
from nn import *
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


#Bayes Classifier Implementation


def class_separation(data):
    #Doing Class Separation (label is in first column)
    #So we are comparing each feature label as it is contained in feature class or not, if not then just append it.
    feature_class={} #Declaring as Dictionary or Set
    for i in range(len(data)):
        feature=data[i]
        if(feature[0] not in feature_class):
            feature_class[feature[0]]=[]
            #feature_class.update({feature[0]:[]})
        feature_class[feature[0]].append(feature)
        
    #print(feature_class)
    #feature_class.pop(3)
    return feature_class

    

def class_summarize(data):
    feature_class = class_separation(data)
    #print(len(feature_class))
    summaries = {}
    for classValue, instances in feature_class.items():
        #Calculate man and standard deviation for each attribute in class
        #Important Step ..... (mean,standard_deviation)
        #summarize = [attribute for attribute in zip(*instances)]
        #print(summarize)
        summarize = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*instances)]
        del summarize[0]
        #print(summarize)
        summaries[classValue] = summarize
    #print(summaries)     
    return summaries

def prior_prob(trainingset):
    a=[]
    u=[]
    prior=[]
    for i in range(len(trainingset)):
        a.append(trainingset[i][0])
    u=np.unique(a)
    u=np.sort(u)
    for i in range(len(u)):
        prior.append(a.count(u[i])/len(a))
    #print(prior)
    return prior
    
 
def calculateProbabilities(summaries, testset,prior): 
    probabilities = {}
    s=[]
    t=[]
    #print(testset)
    #print(prior)
    #print("summa")  
    for classValue, classSummaries in summaries.items():
        #print(classValue)
        #t.append(classValue)
        #print("------")
        #print(classSummaries)
        probabilities[classValue] = 1
        #print(probabilities[classValue])
        #print(len(classSummaries))
        for p in range(0, len(classSummaries)):
            mean, sd = classSummaries[p]
            x = testset[p]
            #print(x)
            #print(sd)
            #To calculate Probability
            if(sd==0):
                #print("Prior"+str(prior[int(classValue)]))
                probabilities[classValue] = probabilities[classValue] * prior[int(classValue)] #Class ka prior find krna hai          
            else:
                #print("Prior"+str(prior[int(classValue)]))
                exponent = math.exp(-(math.pow(x-mean,2))/(2*math.pow(sd,2))) #Exponent Part Calculation
                pr=(1 / ((math.sqrt(2*math.pi) * sd))) * exponent
                #print("pr="+str(pr))
                probabilities[classValue] =probabilities[classValue] * pr
                #print("HI: "+ str(probabilities[classValue]))
    
    #print("End of Class Summary")
    #print("PC"+str(probabilities))
    return probabilities

def predict(summaries, testset,prior):
    probabilities = calculateProbabilities(summaries,testset,prior)
    bestLabel, bestProb = None, -1
    for key, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = key
    return bestLabel
    

def predictit(summary_data,testset,prior):
    predictions=[] #To store predicted values in this array
    testset=np.delete(testset, (0), axis=1)
    for i in range(len(testset)):
        result = predict(summary_data, testset[i],prior)
        predictions.append(result)
    return predictions

def getaccuracy_bayes(testset,prediction):
    correct=0
    for x in range(len(testset)):
        if (testset[x]==prediction[x]):
            correct+=1
    return (correct/float(len(testset)))*100.0 
  
def bayes(data,label):
    start = timeit.default_timer()
    split=0.8
    rand=False
    trainingset=[]
    testset=[]
    t=[]
    p=[]
    data=np.column_stack([label, data])
    #print(len(data))
	
    '''for i in range(len(data)):
	    if(data[i][0]==3):
	        data=np.delete(data, (i), axis=0)
	print(len(data))'''


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

    #print("Number of Training Set: "+str(len(trainingset)))
    #print("Number of Test Set: "+str(len(testset)))
    prior=prior_prob(trainingset)
    feature_class=class_separation(trainingset)
    print("### Bayes Classifier")
    print("Number of Class in Training Set :: "+ str(len(feature_class)))
    summaries=class_summarize(trainingset)

    #Time to Predict
    #print("Running... Please wait")
    predictions = predictit(summaries, testset,prior)
    #print(predictions)
    for x in range(len(testset)):
        t.append(testset[x][0])
        p.append(predictions[x])

    #print(p) #Prediction Array
    #print(t) # Test Value Array
    #Calculating Accuracy
    accuracy = getaccuracy_bayes(t, p)
    print("Test Accuracy :: "+str(round(accuracy,2)))
    print("Test Macro F1-score :: "+ str(round((sklearn.metrics.f1_score(t, p, average="macro")*100),2)))
    print("Test Micro F1-score :: "+ str(round((sklearn.metrics.f1_score(t, p, average="micro")*100),2)))


def bayes_sklearn(data,label):
	from sklearn import metrics
	from sklearn import datasets
	from sklearn.naive_bayes import MultinomialNB
	X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20) 
	#Create a Gaussian Classifier
	model = MultinomialNB()
	# Train the model using the training sets
	model.fit(X_train, y_train)
	#Predict the response for test dataset
	y_pred = model.predict(X_test)
	# Model Accuracy, how often is the classifier correct?
	#print("### Bayes Classifier Using sklearn")
	print("Test Accuracy ::",round(metrics.accuracy_score(y_test, y_pred)*100,2))
	print("Test Macro F1-score :: "+ str(round((sklearn.metrics.f1_score(y_test, y_pred, average="macro")*100),2)))
	print("Test Micro F1-score :: "+ str(round((sklearn.metrics.f1_score(y_test, y_pred, average="micro")*100),2)))




def bayes_sklearn_G(data,label):
    from sklearn import metrics
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20) 
    #Create a Gaussian Classifier
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    #print("### Bayes Classifier Using sklearn")
    print("Test Accuracy ::",round(metrics.accuracy_score(y_test, y_pred)*100,2))
    print("Test Macro F1-score :: "+ str(round((sklearn.metrics.f1_score(y_test, y_pred, average="macro")*100),2)))
    print("Test Micro F1-score :: "+ str(round((sklearn.metrics.f1_score(y_test, y_pred, average="micro")*100),2)))
