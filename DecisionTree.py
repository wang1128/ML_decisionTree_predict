__author__ = 'Penghao Wang'
from numpy import *
from math import *
import csv
#load data in traindata
#load label in the trainlabel
traindata = loadtxt('training.csv',usecols=range(9),dtype=int)
trainlabel = loadtxt('training.csv',usecols=(range(9,10)),dtype=int)

print(traindata)
print(trainlabel)


#calculate entropy
def calculateEntropy(label):
    entropy = 0
    size = label.size
    count = {}
    for currentLabel in label:
        if currentLabel not in count.keys():
            count[currentLabel] = 0
        count[currentLabel] += 1

    for key in count:
        pxi = float(count[key])/size
        entropy -= pxi*log(pxi,2)

    return entropy

x=calculateEntropy(trainlabel)
print(x)

m= [label for label in trainlabel]
featValues = [datasets[0] for datasets in traindata]
uniqueFeatValues = set(m)
print(len(featValues))
print(featValues)

print(uniqueFeatValues)


