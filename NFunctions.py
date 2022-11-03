# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:34:53 2022

@author: Ibrahim
"""



import numpy as np

def relu(x,derivative = False):
    '''
    First, second and Third hidden layer activation function.
    '''
    a = np.array(x > 0, dtype = np.float32) if (derivative== True) else np.maximum(x, 0)
    return a
     
def tanh(x,derivative = False):
    '''
    Second hidden layer activation function.
    '''
    a = (1 -  np.power(np.tanh(x), 2)) if (derivative== True) else np.tanh(x)
    return a 

def sigmoid(x):
    '''
    Output layer activation function.
    '''
    return 1/(1 + np.exp(-x)) 

def initializing(n_x , l1 , l2 , l3, n_y):  # (40000,50,25,10,1)  (input,h1,h2,h3,output)
    '''
    n_x  = neurons in input layer
    n_h1 = neurons in the 1st hidden layer
    n_h2 = neurons in the 2nd hidden layer
    n_h3 = neurons in the 3rd hidden layer
    n_y  = neurons in the output layer
    np.sqrt(1 / n_x)   is xavier initialization
    '''
    #initilization random weights and biases
    w1 =  np.random.randn(l1, n_x) * np.sqrt(1 / n_x)                 #layer 1 x input
    b1 = np.random.rand(l1, 1)                          #layer 1 x bias
    
    w2 = np.random.randn(l2, l1) * np.sqrt(1 / l1)                 #layer x layer 1
    b2 = np.random.rand(l2, 1)                          #layer 2 x bias
    
    w3 = np.random.randn(l3, l2) * np.sqrt(1 / l2)                 #output x layer 2
    b3 = np.random.rand(l3,1) 
    
    w4 = np.random.randn(n_y, l3) * np.sqrt(1 / l3)                #output x layer 2
    b4 = np.random.rand(n_y,1) 
    
    
    return w1, b1, w2, b2, w3, b3, w4, b4     # parameters

def forward_prop(x, w1, b1, w2, b2, w3, b3, w4, b4):
    '''
    Forward propagation and prediction on specific weights
    '''
    
    # first hiden Layer 
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)
    
    # second hiden Layer 
    z2 = np.dot(w2,a1) + b2
    a2 = relu(z2)
    
    # third hiden Layer
    z3 = np.dot(w3,a2) + b3
    a3 = tanh(z3)
    
    # output layer 
    z4 = np.dot(w4,a3) + b4
    a4 = sigmoid(z4)
    
    # forward_cache
    return z1, a1, z2, a2, z3, a3, z4, a4 # forward_cache

def regularization(w1, w2, w3, w4, L, m):
    
    '''
    this function is used for regularization calculation.
    '''
    R = (L/(2*(m))*(np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)) + np.sum(np.square(w4)) ))
    #regularization formula
    return R

def cost_function(a4,y):
    '''
    Finding cost
    
    outputs:
        cost
    '''
    m = y.shape[0]     # Number of examples 
    logprobs =  np.multiply(np.log(a4),y) + np.multiply(np.log(1 - a4),1 - y)
    cost = - np.sum(logprobs) * 1/m  
    cost = float(np.squeeze(cost))                                                  #cost formula
    return cost

def back_prop(x, y, w1, b1, w2, b2, w3, b3, w4, b4, z1, a1, z2, a2, z3, a3, z4, a4):
    
    m = x.shape[0]
    
    # Output Layer
    dz4 = a4-y
    dw4 = (1/ m) * dz4.dot(a3.T)
    db4 = (1/ m) * np.sum(dz4)
    
    # 3rd Hidden Layer
    dz3 = w4.T.dot(dz4) * tanh(z3,derivative = True)
    dw3 = (1/ m) * dz3.dot(a2.T)
    db3 = (1/ m) * np.sum(dz3)
    
    # 2nd Hidden Layer
    dz2 = w3.T.dot(dz3) * relu(z2,derivative = True)
    #print(a1.shape)
    dw2 = (1/ m) * dz2.dot(a1.T)
    db2 = (1/ m) * np.sum(dz2)
    
    # 1st Hidden Layer
    dz1 = w2.T.dot(dz2) * relu(z1,derivative = True)
    dw1 = (1/ m) * dz1.dot(x.T)
    db1 = (1/ m) * np.sum(dz1)
    
    return dz4, dw4, db4, dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1 # gradients

def update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dz4, dw4, db4, dz3, dw3, db3,dz2, dw2, db2, dz1, dw1, db1, learning_rate):
       
    w1 = w1 - learning_rate * dw1
    # print('shape of w1 is',w1.shape)
    b1 = b1 - learning_rate * db1
    
    w2 = w2 - learning_rate * dw2
    #print('shape of w2 is',w2.shape)
    b2 = b2 - learning_rate * db2
    
    w3 = w3 - learning_rate * dw3
    #print('shape of w3 is',w3.shape)
    b3 = b3 - learning_rate * db3
    
    w4 = w4 - learning_rate * dw4    
    #print('shape of w4 is',w4.shape)
    b4 = b4-learning_rate * db4

    
    return w1, b1, w2, b2, w3, b3, w4, b4   # updated parameters


def prediction ( x, y, w1, w2, w3, w4, b1, b2, b3, b4, threshold):
   
    m = x.shape[1]
    # first hiden Layer 
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)
    
    # second hiden Layer 
    z2 = np.dot(w2,a1) + b2
    a2 = relu(z2)
    
    # third hiden Layer
    z3 = np.dot(w3,a2) + b3
    a3 = tanh(z3)
    
    # output layer 
    z4 = np.dot(w4,a3) + b4
    a4 = sigmoid(z4)
    
    #z1, z2, z3, z4, a1, a2, a3, a4 = forward_prop(x, w1, w2, w3, w4, b1, b2, b3, b4)
                                           #updating b1 after back propogation for next iterations
    thres = 0.5                            #setting threshold  
    a4 = np.array(a4)                      #if threshold is < than a3
    a4 = a4 > thres                        #comparing threshold with a3
    y = np.array(y)                        #converting a4 in np array of int64 
    
    tp = np.sum((y == 1) & (a4 == 1))       #calculating true positive
    tn = np.sum((y == 0) & (a4 == 0))       #calculating true negative
    fp = np.sum((y == 0) & (a4 == 1))       #calculating false positive
    fn = np.sum((y == 1) & (a4 == 0))       #calculating false negative
         
    Accuracy = (tp+tn)*(100/m)              #calculating Accuracy
    precision = (tp)/(tp+fp)                #calculating precision
    recall = (tp)/(tp+fn)                   #calculating recall
        
    F1Score = (2*(precision*recall)/(precision+recall))       #calculating F1 Score
    
    
    #print('Number of True Positive :', tp)
    #print('Number of True Negative :', tn)
    #print('Number of False Positive :', fp)
    #print('Number of False Negative :', fn)
    
    
   
    print('Accuracy of the model is :', Accuracy)
    #print('F1 score of the model is :',F1Score )
    return Accuracy,tp,tn,fp,fn, recall, precision,F1Score
