__author__ = 'Penghao Wang'
from numpy import *
from math import *

#load data in traindata
#load label in the trainlabel
traindata = loadtxt('training.csv',usecols=range(9),dtype=int)
trainlabel = loadtxt('training.csv',usecols=(range(9,10)),dtype=int)
validatingdata = loadtxt('validating.csv',usecols=range(9),dtype=int)
validatingLabel = loadtxt('validating.csv',usecols=(range(9,10)),dtype=int)
testdata = loadtxt('testing.csv',usecols=range(9),dtype=int)
testLabel = loadtxt('testing.csv',usecols=(range(9,10)),dtype=int)
feaname = ["#0","#1","#2","#3","#4","#5","#6","#7","#8"]

args = mean(traindata,axis = 0)

class Node:
    count=0
    nodeList=set()
    nodeAttributeidx=[0,1,2,3,4,5,6,7,8]
    nodeAidxSplitValue=[0,0,0,0,0,0,0,0,0]

node= Node()

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

#split the dataset 
def splitdata(data,splitfeature_index,splitvalue):
    index_less = []
    index_more = []

    for index in range(len(data)):
        d = data[index]
        if d[splitfeature_index] < splitvalue:
            index_less.append(index)
        else:
            index_more.append(index)
    return index_less,index_more

def chooseBestSplitNode(data,label):
    best_featureindex=-1
    splitFeatures=-1
    classList = [example for example in trainlabel]

    if size(data) not in range(0,2) and classList.count(classList[0]) != len(classList) :
        n_fea = len(data[0])
        n = len(label)
        base_entropy = calculateEntropy(label)
        best_gain = 0
        for fea_i in node.nodeAttributeidx:
            cur_entropy = 0
            featValues = [datasets[fea_i] for datasets in traindata]
            uniqueFeatValues = list(set(featValues))

            for i in uniqueFeatValues:
                cur_entropy = 0
                idxset_less,idxset_greater = splitdata(data,fea_i,i)

                prob_less = float(len(idxset_less))/n
                prob_greater = float(len(idxset_greater))/n
                cur_entropy += prob_less*calculateEntropy(label[idxset_less])
                cur_entropy += prob_greater * calculateEntropy(label[idxset_greater])
                info_gain = base_entropy - cur_entropy

                if(info_gain>best_gain):
                    best_gain = info_gain
                    best_featureindex = fea_i
                    splitFeatures = i

    return best_featureindex,splitFeatures


#print(secondLeftData)
#print(len(secondLeftData))

def buildTree(traindata,trainlabel):

    if trainlabel.size <= 1:
        return trainlabel[0]
    classList = [example for example in trainlabel]
    # the type is the same, so stop classify
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    curbestixd,cursplitF=chooseBestSplitNode(traindata,trainlabel)
    if curbestixd in node.nodeAttributeidx:
        node.nodeAttributeidx.remove(curbestixd)
    curnode= (curbestixd,cursplitF)
    node.nodeAidxSplitValue[curbestixd]= cursplitF
    if curnode == (-1,-1): return None
    else:
        print(curbestixd,cursplitF)
        cur_feaname = feaname[curbestixd]
        print cur_feaname
        nodedict = {cur_feaname:{}}

        node.count += 1
        if curnode in node.nodeList:
            return
        f = (curnode,node.count)
        node.nodeList.add(f)

        curLeftIdx,curRightIdx=splitdata(traindata,curbestixd,cursplitF)
        curLeftdataSet= traindata[curLeftIdx]
        curLeftdataLabel = trainlabel[curLeftIdx]
        curRightdataSet = traindata[curRightIdx]
        curRightdataLabel = trainlabel[curRightIdx]
    nodedict[cur_feaname]["leftNode"] = buildTree(curLeftdataSet,curLeftdataLabel)
    print("l")
    nodedict[cur_feaname]["RightNode"] = buildTree(curRightdataSet,curRightdataLabel)
    print("r")
    return nodedict



def classify(mytree,testdata):
    if type(mytree).__name__ != 'dict':
        return mytree
    fea_name = mytree.keys()[0] #get the name of first feature
    fea_idx = feaname.index(fea_name) #the index of feature 'fea_name'
    val = testdata[fea_idx]
    nextbranch = mytree[fea_name]

    #judge the current value > or < the pivot (average)
    if val>node.nodeAidxSplitValue[fea_idx]:
        nextbranch = nextbranch["RightNode"]
    else:
        nextbranch = nextbranch["leftNode"]
    return classify(nextbranch,testdata)




mytree=buildTree(traindata,trainlabel)
print(mytree)
print(node.nodeList)
a=0
b=0
c=0
for idx in range(60):
    tt = traindata[idx]
    x = classify(mytree,tt)
    if (x == trainlabel[idx]):
        a+=1

print a

for idx in range(20):
    tt = validatingdata[idx]
    x = classify(mytree,tt)
    if (x == validatingLabel[idx]):
        b+=1

print b



for idx in range(20):
    tt = testdata[idx]
    x = classify(mytree,tt)
    if (x == testLabel[idx]):
        c+=1

print c
