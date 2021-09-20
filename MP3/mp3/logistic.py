import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    m = 0
    X = np.mat(X)

    # YOUR CODE HERE
    # begin answer
    # TODO
    # end answer
    
    while m < 50:
        m += 1
        for i in range(N):
            x = X[:,i].transpose()
            x = np.c_[x, np.array([1])]
            result = np.dot(x,w)
            result = 1/(1 + np.exp(-result))
            k = np.array((y[:,i] - result))
            w += np.multiply(np.multiply(np.multiply((1/m)*k, result), 1-result),x.transpose())   

    # print(w)
    a = w[2]
    w = np.delete(w,2)
    w = np.r_[a, w]
    w = np.array([[w[0]],[w[1]],[w[2]]])
    # print(w)

    return w
