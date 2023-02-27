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
        inner_prod = np.dot(X, self.w)
        loss = 0
        loss = np.sum(np.log(1 + np.exp(inner_prod))-Y*inner_prod)
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
        inner_prod = np.dot(X, self.w)
        gradients = (np.exp(inner_prod)/(1+np.exp(inner_prod)) - Y).T@X
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
            self.w = np.sum([self.w, -eta*gradient], axis = 0)


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