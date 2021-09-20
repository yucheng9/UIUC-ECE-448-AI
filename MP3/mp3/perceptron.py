import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    X = np.mat(X)
    # YOUR CODE HERE
    # begin answer
    # TODO
    # end answer
    while True:
      iters += 1
      m = 0
      for i in range(N):
        x = X[:,i].transpose()
        x = np.c_[x, np.array([1])]
        result = np.dot(x,w)
        result = np.sign(result)
        if result != y[:,i]:
            w += np.multiply(((1/(iters))) * np.array(y[:, i]) , x.transpose())
            m = 1
      if m == 0:
        break
    # print(w)
    a = w[2]
    w = np.delete(w,2)
    w = np.r_[a, w]
    # print(w)
        
    return w, iters