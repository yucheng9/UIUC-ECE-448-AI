import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    batch_size = 200
    num_batches = int(len(x_train)/batch_size)
    losses = [0 for x in range(epoch)]

    for i in range(epoch):

        print("Epoch: ", i+1)
        loss = 0

        if (shuffle == True):
            idx = np.random.choice(len(x_train), len(x_train), False)
            x = x_train[idx]
            y = y_train[idx]

        else:
            x = x_train.copy()
            y = y_train.copy()

        for j in range(num_batches):
            x_test = x[j*batch_size : (j+1)*batch_size]
            y_test = y[j*batch_size : (j+1)*batch_size]
            loss += four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, False)
        
        losses[i] = loss
    
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    class_rate_per_class = [0.0] * num_classes
    classifications = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, True)
    avg_class_rate = np.sum(y_test == classifications)/len(y_test)

    for i in range(num_classes):
        class_rate_per_class[i] = np.sum(classifications[np.argwhere(y_test == i)] == i)/len(np.argwhere(y_test == i))

    return avg_class_rate, class_rate_per_class,classifications

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, test):
    Z1, acache1 = affine_forward(x_test, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w2, b2)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, w4, b4)

    if test:
        return np.argmax(F, axis=1)

    else:
        eta = 0.1
        loss, dF = cross_entropy(F, y_test)

        dA3, dW4, db4 = affine_backward(dF, acache4)
        dZ3 = relu_backward(dA3, rcache3)
        dA2, dW3, db3 = affine_backward(dZ3, acache3)
        dZ2 = relu_backward(dA2, rcache2)
        dA1, dW2, db2 = affine_backward(dZ2, acache2)
        dZ1 = relu_backward(dA1, rcache1)
        dX, dW1, db1 = affine_backward(dZ1, acache1)

        w1 -= eta*dW1
        w2 -= eta*dW2
        w3 -= eta*dW3
        w4 -= eta*dW4
        b1 -= eta*db1
        b2 -= eta*db2
        b3 -= eta*db3
        b4 -= eta*db4

        return loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    return np.matmul(A,W)+b, (A, W)

def affine_backward(dZ, cache):
    dA = np.matmul(dZ, cache[1].T)
    dW = np.matmul(cache[0].T, dZ)
    dB = np.sum(dZ, axis=0)
    return dA, dW, dB

def relu_forward(Z):
    A = Z.copy()
    A[A<0] = 0
    return A, Z

def relu_backward(dA, cache):
    dA[np.where(cache<0)] = 0
    return dA

def cross_entropy(F, y):
    loss = -(1/len(F))*np.sum(F[np.arange(len(F)), y.astype(int)] - np.log(np.sum(np.exp(F), axis=1)))
    class_label = np.zeros(F.shape)
    class_label[np.arange(len(F)), y.astype(int)] = 1  
    dF = -(1/len(F))*(class_label - np.exp(F) / (np.sum(np.exp(F), axis=1)).reshape((-1, 1)) )

    return loss, dF
