import numpy as np
import matplotlib.pyplot as plt

class Polynomial:

    def __init__(self, coeff):
        self.coefficients = np.array(coeff)
    
    def __call__(self, x):
        ret = 0
        for idx, coeff in enumerate(self.coefficients[::-1]):
            ret += coeff*(x**idx)
        return ret
    
    def get_grad(self, x):
        ret = 0
        for idx, coeff in enumerate(self.coefficients[::-1]):
            ret += coeff*idx*(x**(idx-1))
        return ret
    
    def get_min_idx(self):
        return -self.coefficients[1]/(2*self.coefficients[0])

    
    
def convex_function_generate(n):
    ret = []
    for _ in range(n):
        coeff = np.random.randint(1,10, size=3)
        ret.append(Polynomial(coeff))
    return ret

class Environment:

    def __init__(self, N=50):
        self.n = N
        self.fts = convex_function_generate(self.n)
        
        # For debugging, fixed case
        # self.fts = [Polynomial([1,6,9])]
        # self.n=1

        self.loss_fts = {}

    def choose_loss_ft(self, t):
        self.loss_fts[t] = self.fts[np.random.randint(self.n)]

    def get_loss_val(self, t, x):
        assert self.loss_fts[t] != None
        return self.loss_fts[t](x)
    
    def get_loss_ft(self,t):
        assert self.loss_fts[t] != None
        return self.loss_fts[t]

    def get_regret(self,t,x):
        assert self.loss_fts[t] != None
        opt_x = self.loss_fts[t].get_min_idx()
        return self.loss_fts[t](x)-self.loss_fts[t](opt_x)




# if __name__ == "__main__":
    # p = Polynomial([1,-8,2])
    # x = np.linspace(-10,10,1000)
    # y = p(x)
    # plt.plot(x, y)
    # plt.savefig("ex.png")