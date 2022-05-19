import json
import struct
import numpy as np   # check out how to install numpy
from utils import load, plot_sample, euclidean_distance
from sklearn.metrics import precision_score, accuracy_score


# =========================================
#       Homework on K-Nearest Neighbors
# =========================================


Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'
'''
sampleNum = 8
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])
plot_sample(Xtrain[sampleNum, :], Ytrain[sampleNum, :])
'''
# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.
#calculating the destance between each raw in Xvalid with Xtrain


def predictor(vec,Xtra,k,ref_vec):#this function calssifies the pictures:given picture "vec", we deside wither its a 5 of a 3..
    distances = np.linalg.norm(Xtra - vec,axis = 1) #vector of distances between each matrix row and a given vector vec
    indices_of_k_samallest =sorted(range(len(distances)), key=lambda sub: distances[sub])[:k]
    test_vec=[]
    for j in indices_of_k_samallest:
        test_vec.append(np.array(ref_vec).flatten()[j])

    if np.count_nonzero(np.array(test_vec).flatten()==3) > np.count_nonzero(np.array(test_vec).flatten()==5):
        return 3
    else:   # I've chose an odd K
        return 5

def labeling(mat, ref_vec, Xtra, k): #this function gives a prediction vector of Xvalid pictures
    Ypred = []
    for i in range(mat.shape[0]):
        Ypred.append(predictor(mat[i],Xtra,k,ref_vec))
    return np.array(Ypred).flatten()

def accuracy(true,pred): #calculating accuracy
    return np.equal(true, pred).mean()
'''
accuracy_vec = []
a_0=11
for x in range(a_0,37, 2): #k=11,13,15...,35.
    accuracy_vec.append(accuracy(np.array(Yvalid).flatten(),labeling(Xvalid,Ytrain,Xtrain,x)))
result = np.array(accuracy_vec).argmax() # index of the opteman accuracay
k_opt=a_0+2*(result-1) # using the  n-th term of arithmetic sequence: an= a0+d*(n-1)
print(f'the accuracy vector is:{accuracy_vec}')
print(f'the optimal accuracy is{accuracy_vec[result]}')
print(f'the idael K is {k_opt}')



'''
#read me:
# in the commented code, i've iterated over some values of k(i found on internet that it should be odd ).
# at the beggining i chose a random k to work with, then i i asek a friend i which intervals should i iterte.
# so lastly, i ran over : 11,13,15...,35.
#i've commented the procces of getting the optimal k, because it toke me about 30 mintes to finish.
# i got k=15 is the optimal with 0.99 accuracy.
#here is the rest of the code where i generate Ytest.
k_opt=15

Ytest=labeling(Xtest,Ytrain,Xtrain,k_opt)

# Example submission array - comment/delete it before submitting:
#Ytest = np.arange(0, Xtest.shape[0])

# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')
