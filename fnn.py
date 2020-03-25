import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(n_y, n_h)
    b2 = np.random.randn(n_y, 1)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#feedforward
def linear_activation_forward(A_prev, W, b):
    for bi, w in zip(b,W):
        A = sigmoid(np.dot(W,A_prev)+b)
    return A

#sigmoid
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#one-hot vector for labels
def onehot(label):
    E = np.zeros((7))
    E[label] = 1.0
    return E

#compute cost
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    #cross entropy
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = np.sum(logprobs)/m
    
    cost = np.squeeze(cost)
    
    assert(isinstance(cost, float))
    return cost

#back propogation
def back_propogation(parameters, A1, A2, X, Y):
    m = X.shape[0]
    print(m)
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    dZ2 = A2 - Y
    dW2 = (1/m)* np.dot(dZ2, A1.T)
    
    print(dW2.shape)
    
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2), 1 - np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}
    
    return grads


#update parameters
def update_parameters(parameters, grads, learning_rate=0.25):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters



#accuracy
def accuracy(data, parameters, label ):
    index = np.argmax(Y, axis=0)
    correct = 0
    for i in range(10000):
        if (index[i] == label[i]):
            correct += 1
    
    acc = correct/10000
    print("Accuracy : ",str(acc*100))
    
    
if __name__ == "__main__":
    #load fer
    with open("fer_dataset/train.csv") as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    data = []
    labels = []
    for i in range(1, num_of_instances):
        try:
            emotion, img = lines[i].split(",")
            img = img.replace('"', '')
            img = img.replace('\n', '')
            pixels = img.split(" ")
            
            data.append(pixels)
            labels.append(emotion)

        except Exception as ex:
            print(ex)
    
    labels = np.asarray(labels, dtype=np.int32)
    data = np.asarray(data, dtype=np.int32)
    print(data.shape)
    print(labels.shape)
    
    
    #load test fer
    with open("fer_dataset/test.csv") as f:
        test_content = f.readlines()

    test_lines = np.array(test_content)

    test_num_of_instances = test_lines.size
    test_data = []
    test_labels = []
    for i in range(1, test_num_of_instances):
        try:
            emotion, img = lines[i].split(",")
            img = img.replace('"', '')
            img = img.replace('\n', '')
            pixels = img.split(" ")
            
            test_data.append(pixels)
            test_labels.append(emotion)

        except Exception as ex:
            print(ex)
    
    test_labels = np.asarray(test_labels, dtype=np.int32)
    test_data = np.asarray(test_data, dtype=np.int32)
    print(test_data.shape)
    print(test_labels.shape)
    
    
    
    one_hot_vectors = [onehot(y) for y in labels]
    one_hot_vectors = np.asarray(one_hot_vectors)
    print(one_hot_vectors[0])

    parameters = initialize_parameters(2304,700,7)
    cost = 0
    for i in range(100):
        datat = data.T
        print("-------------------------------------")
        print('iteration = ', str(i+1))
        
        X = linear_activation_forward(datat, parameters['W1'], parameters['b1'])
        
        Y = linear_activation_forward(X, parameters['W2'], parameters['b2'])
        
        index = np.argmax(Y, axis=0)
        
        cost = compute_cost(Y, one_hot_vectors, parameters)
        print("Cost = ", str(cost))
        
        grads = back_propogation(parameters, X, Y, data, one_hot_vectors)
        
        parameters = update_parameters(parameters, grads)
        
        accuracy(Y, parameters, labels)
        
    
    print(Y.shape)
    accuracy(data,parameters,labels)
