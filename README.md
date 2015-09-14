# Penghao Wang homework 1
##Goals
The program is written in Python 2.7. The numpy liabray is needed. The main purpose of this project is to build a decision tree by using traindata.csv and prune the tree by
validating.csv

##Instructions
 - I split the original data (training.csv and validating.csv) into two parts: data (1-9 columns) and the labels(10 columns)
 -  You need to import the files into the traindata, trainlabel, validatingdata, validatingLabel, testdata and testLabel.
  The project include these .csv files in the folder. You need to import files by using correct directory path
 - The calculateEntropy() is to calculate Entropy of every values in every attrubites.
 - The splitdata() is to split the dataset by using splitFeature and splitValue. It return the index_less and index_more of dataset 
 - The chooseBestSplitNode() is help to choose best split node. In the function I calculate the best Infromation Gain of each values in each Attributes. I use a binary tree as a decision tree.
 The values that less than split value go to the left of the node.
 - The buildTree() function is build the tree by using the data and labels of the data.
 - The classify() function is to use the tree to classfity the data and return the binary label (columns 10)
 - The counAcc() function is to count the number of right labels that are the same as original data
 - The prune() function is to prune the tree by using the validating data
 - To run the program, you need to bulid a tree. For examples: myTree=buildTree(traindata,trainlabel)
 - Then use prune() to prune the tree. For instance: pruneTree = prune(validatingdata,validatingLabel,myTree)
 - Finally, use countAcc() to count the accuracy
 
##Requirements:
 - Python 2.7
 - Import numpy

