
# coding: utf-8


import nltk
import csv
import math
import sys
import numpy as np
import tensorflow as tf
from nltk.data import path
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, regexp_tokenize
from random import random

#Training Xs and Ys. X presents array of tweets and y are polarities of tweets(1 positive, 0 negative)
X, y = [], []

#Test sets X and Y
Xtest, ytest = [], []

#Validation sets X and Y
Xval, yval = [], []

#Dictionary that will contain all words from tweets and their value. Value is polarity of word(how positive[or negative] word is)
my_dict = {}

#First we're opening training set and splitting data in trainig-validating-test sets(60-20-20).
with open('train.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader, None)  # Skip header
    for row in reader:
        #if random() < 0.1:
        if random() < 0.6:
            y.append(int(row[1]))
            X.append(row[2].lower())
        else:
            if random() < 0.5:
                yval.append(int(row[1]))
                Xval.append(row[2].lower())
            else:
                ytest.append(int(row[1]))
                Xtest.append(row[2].lower())

            # print VAL
print(len(Xtest), ' ', len(X), ' ', len(Xval))

################# Processing training set ####################
# Here we're removing non-words from tweets and stopwords. Then we're adding words in dictionary.
# With each appearance of word that already exists in dictionary we're adding or subtracting 1 to/from the value of word.
for i in range(len(X)):
    words = regexp_tokenize(X[i], "[\w']+")

    stopword_list = set(stopwords.words('english'))
    words_filtered = [w for w in words if w not in stopword_list]

    final = []
    lancaster = LancasterStemmer()
    for word in words_filtered:
        final.insert(len(final), lancaster.stem(word))
        if final[len(final) - 1] in my_dict:
            if y[i] == 1:
                my_dict[final[len(final) - 1]] += 1
            else:
                my_dict[final[len(final) - 1]] -= 1
        else:
            if y[i] == 1:
                my_dict[final[len(final) - 1]] = 1
            else:
                my_dict[final[len(final) - 1]] = -1

    X[i] = final

print('Preprocesing training set finished')
################# END processing training set ####################


################# Processing validation set ####################
# Similar for validation set, except for values part.

print('Preprocesing validation set')
for i in range(len(Xval)):
    words = regexp_tokenize(Xval[i], "[\w']+")

    stopword_list = set(stopwords.words('english'))
    words_filtered = [w for w in words if w not in stopword_list]

    final = []
    lancaster = LancasterStemmer()
    for word in words_filtered:
        final.insert(len(final), lancaster.stem(word))
    Xval[i] = final

print('Preprocesing validation set finished')
################# END processing validation set ####################

################# Processing test set ####################
# Similar for test set as for validation set.
print('Preprocesing test set')
for i in range(len(Xtest)):
    words = regexp_tokenize(Xtest[i], "[\w']+")

    stopword_list = set(stopwords.words('english'))
    words_filtered = [w for w in words if w not in stopword_list]

    final = []
    lancaster = LancasterStemmer()
    for word in words_filtered:
        final.insert(len(final), lancaster.stem(word))
    Xtest[i] = final

print('Preprocesing test set finished')
################# END processing test set ####################


#Positive training values, Negative training values
Ptr, Ntr = [0] * len(X), [0] * len(X)

#Positive validation values, Negative validation values
Pval, Nval = [0] * len(Xval), [0] * len(Xval)

#Positive testing values, Negative testing values
Pts, Nts = [0] * len(Xtest), [0] * len(Xtest)

print('Start calculating 2d positions of tweets')
# Calculating 2d position of every tweet in training samples
for i in range(len(X)):
    for word in X[i]:
        if word in my_dict:
            if my_dict[word] > 0:
                Ptr[i] += 1
            else:
                Ntr[i] += 1

# Calculating 2d position of every twet in validation samples
for i in range(len(Xval)):
    for word in Xval[i]:
        if word in my_dict:
            if my_dict[word] > 0:
                Pval[i] += 1
            else:
                Nval[i] += 1

# Calculating 2d position of every twet in test samples
for i in range(len(Xtest)):
    for word in Xtest[i]:
        if word in my_dict:
            if my_dict[word] > 0:
                Pts[i] += 1
            else:
                Nts[i] += 1

print('End calculating 2d positions of tweets')


# Accuracy for algorithm.
acc = 0.

# KNN algorithm is used, so this is the K.
k = 3

# Accuracy value for validation
acc_val = 0.

# Best K that's got from validation
best_k = 0

# Sigmoid function of Weighted K Nearest Neighbors algorithm
def sigma(c, b):
    if c == b:
        return 1
    else:
        return 0
    
# The biggest float weight for calculating our weights - used if distance between two tweets(points) is 0.
# Because it means that these tweets are the same polarity.
big_f_weight = sys.float_info.max/1000;
    
############################## VALIDATION TO FIND BEST_K ON VALIDATION SET ###################################################

print('------------------- VALIDATION ------------------------')

for i in range(len(Pval)):
    
    # At this epoch we're incrementing K and checking the best value of K for the best accuracy
    if (i % (len(Pval)//7))==0:
        if acc_val < acc:
            acc_val = acc
            best_k = k
            
        print(' ')
        print('epoch: ', i ,';current acc: ', acc, ';best acc: ', acc_val, ';current k: ', k, ';best k: ', best_k)
        print(' ')
        
        acc = 0.
        k += 2
        
    # Distances between tweets
    dist = []
    
    # Weights
    weights = []

    # Calculating distances and weights.
    for j in range(len(Ptr)):
        dist.append(math.sqrt(math.pow(Ptr[j] - Pval[i], 2) + math.pow(Ntr[j] - Nval[i], 2)))
        if dist[len(dist)-1] == 0:
            weights.append(big_f_weight)
        else:
            weights.append(1/dist[len(dist)-1])
        
    # Use of numpy 
    np_dist = np.array(dist)
    
    # Returns indices of k smallest distances
    k_smallest = np_dist.argsort()[:k]
    
    # Counters of tweets in class 1 and class 0 
    cls_cnt1 = 0
    cls_cnt0 = 0
    
    # KNN algorithm
    for j in range(len(k_smallest)):
        if y[k_smallest[j]] == 1:
            cls_cnt1 += weights[k_smallest[j]] * sigma(1, y[k_smallest[j]])
        else:
            cls_cnt0 += weights[k_smallest[j]] * sigma(0, y[k_smallest[j]])

    
    cls = 1 if cls_cnt1 > cls_cnt0 else 0
    
    # Checking if we got a match!
    match = False

    if cls == yval[i]:
        acc += 1.0 / (len(yval)//7)
        match = True

    if i % 500 == 0:
        print("[Test %3d] Prediction: %d, True Class: %d, Match: %d" % (i, cls, yval[i], match))

print("Validation training accuracy: ", acc_val)

#val_test_acc = acc_val

acc = 0.
acc_val = 0.0

############################## TESTING FOR BEST_K ON VALIDATION SET ###########################################################

print('------------------- VALIDATION TESTING ------------------------')

for i in range(len(Pval)):
    
    dist = []
    weights = []

    for j in range(len(Ptr)):
        dist.append(math.sqrt(math.pow(Ptr[j] - Pval[i], 2) + math.pow(Ntr[j] - Nval[i], 2)))
        if dist[len(dist)-1] == 0:
            weights.append(big_f_weight)
        else:
            weights.append(1/dist[len(dist)-1])
        
    np_dist = np.array(dist)
    #Returns indices of k smallest distances
    k_smallest = np_dist.argsort()[:best_k]
    cls_cnt1 = 0
    cls_cnt0 = 0
    
    for j in range(len(k_smallest)):
        if y[k_smallest[j]] == 1:
            cls_cnt1 += weights[k_smallest[j]] * sigma(1, y[k_smallest[j]])
        else:
            cls_cnt0 += weights[k_smallest[j]] * sigma(0, y[k_smallest[j]])

    
    cls = 1 if cls_cnt1 > cls_cnt0 else 0
    
    match = False

    if cls == yval[i]:
        acc_val += 1.0 / len(yval)
        match = True

    if i % 500 == 0:
        print("[Test %3d] Prediction: %d, True Class: %d, Match: %d" % (i, cls, yval[i], match))

print("Validation Accuracy: ", acc_val)


print('------------------- FINAL TESTING ------------------------')
############################## TESTING FOR BEST_K ON TEST SET ###############################################################
acc = 0.
for i in range(len(Pts)):
    
    dist = []
    weights = []

    for j in range(len(Ptr)):
        dist.append(math.sqrt(math.pow(Ptr[j] - Pts[i], 2) + math.pow(Ntr[j] - Nts[i], 2)))
        if dist[len(dist)-1] == 0:
            weights.append(big_f_weight)
        else:
            weights.append(1/dist[len(dist)-1])
        
    np_dist = np.array(dist)
    #Returns indices of k smallest distances
    k_smallest = np_dist.argsort()[:best_k]
    cls_cnt1 = 0
    cls_cnt0 = 0
    
    for j in range(len(k_smallest)):
        if y[k_smallest[j]] == 1:
            cls_cnt1 += weights[k_smallest[j]] * sigma(1, y[k_smallest[j]])
        else:
            cls_cnt0 += weights[k_smallest[j]] * sigma(0, y[k_smallest[j]])

    
    cls = 1 if cls_cnt1 > cls_cnt0 else 0
    
    match = False

    if cls == ytest[i]:
        acc += 1.0 / len(ytest)
        match = True

    if i % 500 == 0:
        print("[Test %3d] Prediction: %d, True Class: %d, Match: %d" % (i, cls, ytest[i], match))

print("TESTING Accuracy: ", acc)

# Printing the accuracy of whole testing data and validating data
print('test: ', acc, ' val: ', acc_val)


# In[ ]:




