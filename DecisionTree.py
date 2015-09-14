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

class Node:
    count=0
    nodeList=set()
    nodeAttributeIdx=[0,1,2,3,4,5,6,7,8] #defalut the Attribute Index
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

#split the dataset by using splitFeature and splitValue
#return the index_less and index_more of data
def splitdata(data,splitfeature_index,splitvalue):
    index_less = []
    index_more = []

    for index in range(len(data)):
        d = data[index]
        #if the value in the attributes less than split value that was choosed then add to index_less
        if d[splitfeature_index] < splitvalue:
            index_less.append(index)
        else:
            index_more.append(index)
    return index_less,index_more

#chooseBestSplitNode by using Information Gain
def chooseBestSplitNode(data,label):
    #defalut values
    best_featureindex=-1
    splitFeatures=-1
    classList = [example for example in trainlabel]

    if size(data) not in range(0,2) and classList.count(classList[0]) != len(classList) :
        n_fea = len(data[0])
        n = len(label)
        base_entropy = calculateEntropy(label)
        best_gain = 0
        for fea_i in node.nodeAttributeIdx:
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

#build the Tree by using splitFeature and splitValue

def buildTree(traindata,trainlabel):
    #if there is only one value then return the value
    if trainlabel.size <= 1:
        return trainlabel[0]
    classList = [example for example in trainlabel]
    #check whether the whole labels are same or not
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    curbestixd,cursplitF=chooseBestSplitNode(traindata,trainlabel)
    if curbestixd in node.nodeAttributeIdx:
        node.nodeAttributeIdx.remove(curbestixd)
    curNode= (curbestixd,cursplitF)
    node.nodeAidxSplitValue[curbestixd]= cursplitF
    if curNode == (-1,-1): return None
    else:
        cur_feaname = feaname[curbestixd]
        #use dictionary to save the Tree
        nodeDict = {cur_feaname:{}}

        node.count += 1
        if curNode in node.nodeList:
            return
        f = (curNode,node.count)
        node.nodeList.add(f)

        curLeftIdx,curRightIdx=splitdata(traindata,curbestixd,cursplitF)
        curLeftdataSet= traindata[curLeftIdx]
        curLeftdataLabel = trainlabel[curLeftIdx]
        curRightdataSet = traindata[curRightIdx]
        curRightdataLabel = trainlabel[curRightIdx]
    nodeDict[cur_feaname]["leftNode"] = buildTree(curLeftdataSet,curLeftdataLabel)
    nodeDict[cur_feaname]["RightNode"] = buildTree(curRightdataSet,curRightdataLabel)
    return nodeDict

def classify(myTree,testdata):
    if type(myTree).__name__ != 'dict':
        return myTree
    fea_name = myTree.keys()[0]
    fea_idx = feaname.index(fea_name)
    val = testdata[fea_idx]
    nextNode = myTree[fea_name]

    if val>node.nodeAidxSplitValue[fea_idx]:
        nextNode = nextNode["RightNode"]
    else:
        nextNode = nextNode["leftNode"]
    return classify(nextNode,testdata)

def countAcc (data,label,myTree):
    countRight = 0
    for idx in range(len(data)):
        t = data[idx]
        x = classify(myTree,t)
        if (x == label[idx]):
            countRight+=1

    return countRight

#prune the Tree by using the validating Data
def prune (validatingdata,validatingLabel,myTree):

    node.nodeAttributeIdx = [0,1,2,3,4,5,6,7,8]
    maxRightNum= countAcc(validatingdata,validatingLabel,myTree)

    pruneNodeIdx = 0
    for nodeIdx in node.nodeAttributeIdx:
        node.nodeAttributeIdx = [0,1,2,3,4,5,6,7,8]
        node.nodeAttributeIdx.remove(nodeIdx)
        proneTree = buildTree(traindata,trainlabel)
        rightValueCount=0
        for idx in range(20):

            test = validatingdata[idx]
            x = classify(myTree,test)
            if (x == validatingLabel[idx]):
                rightValueCount+=1
        if maxRightNum< rightValueCount:
            maxRightNum = rightValueCount
            pruneIdx = nodeIdx
        else:
            pruneIdx = -1

    node.nodeAttributeIdx = [0,1,2,3,4,5,6,7,8]
    #if there is no need to prune the tree then not
    if pruneIdx != -1:
        node.nodeAttributeIdx.remove(pruneIdx)
    proneTree = buildTree(traindata,trainlabel)
    return proneTree

def main():
    myTree=buildTree(traindata,trainlabel)
    pruneTree = prune(validatingdata,validatingLabel,myTree)
    testAccValue=countAcc(testdata,testLabel,pruneTree)
    #print(pruneTree)
    print(testAccValue) #It calculate how many of the classified data is the right ones.
    testAccRate = 0.00
    testAccRate = testAccValue.__float__()/len(testLabel).__float__()

    print("The accuracy rate of the tree by using the test data is:",testAccRate )

main()
