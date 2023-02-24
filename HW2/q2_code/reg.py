import numpy as np

class LogisticRegression:
    def __init__(self, d):
        self.w = np.random.randn(d)

    def compute_loss(self, X, Y):
        """
        Compute l(w) with n samples.
        Inputs:
            X  - A numpy array of size (n, d). Each row is a sample.
            Y  - A numpy array of size (n,). Each element is 0 or 1.
        Returns:
            A float.
        """
        bias = 0
        inner_prod = np.zeros(X.shape[0])
        for idx0 in range(X.shape[0]):
            for idx1 in range(X.shape[1]):
                inner_prod[idx0] += self.w[idx1]*X[idx0, idx1]
        loss = 0
        for idx0 in range(X.shape[0]):
            loss += np.log(1 + np.exp(inner_prod[idx0]))-Y[idx0]*inner_prod[idx0]
        loss = loss/X.shape[0]
        return loss
        
    def compute_grad(self, X, Y):
        """
        Compute the derivative of l(w).
        Inputs: Same as above.
        Returns:
            A numpy array of size (d,).
        """
        bias = 0
        gradients = np.zeros(X.shape[1])
        # The sufficient statistic for conditional MLE
        inner_prod = [0 for i in range(X.shape[0])]
        for idx0 in range(X.shape[0]):
            for idx1, x_val in enumerate(X[idx0, :]):
                inner_prod[idx0] += self.w[idx1]*x_val

        for idx0 in range(X.shape[1]):
            for idx1 in range(X.shape[0]):
                  gradients[idx0] += X[idx1,idx0]*(
                    (np.exp(inner_prod[idx1]))/(1+np.exp(inner_prod[idx1]))- Y[idx1]
                    )      
        gradients = gradients/X.shape[0]
        return gradients

    def train(self, X, Y, eta, rho):    
        """
        Train the model with gradient descent.
        Update self.w with the algorithm listed in the problem.
        Returns: Nothing.
        """
        while True:
            gradient = self.compute_grad(X, Y)
            magnitude = np.linalg.norm(gradient)
            if magnitude < rho:
                break
            self.w = np.sum([self.w, -gradient], axis = 0)


if __name__ == '__main__':
    # Sample Input/Output
    d = 10
    n = 1000

    np.random.seed(0)
    X = np.random.randn(n, d)
    Y = np.array([0] * (n // 2) + [1] * (n // 2))
    eta = 1e-3  
    rho = 1e-6

    reg = LogisticRegression(d)
    reg.train(X, Y, eta, rho)
    print(reg.w)
    print(reg.compute_loss(X,Y))
    # The output should be close to
    # [ 0.15289573 -0.063752   -0.06434498 -0.02005378  0.07812127 -0.04307333
    #  -0.0691539  -0.02769485 -0.04193284 -0.01156307]
    # Error should be less than 0.001 for each element