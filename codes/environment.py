import numpy as np
import matplotlib.pyplot as plt

# class Polynomial:

#     def __init__(self, coeff):
#         self.coefficients = np.array(coeff)
    
#     def __call__(self, x):
#         ret = 0
#         for idx, coeff in enumerate(self.coefficients[::-1]):
#             ret += coeff*(x**idx)
#         return ret
    
#     def get_grad(self, x):
#         ret = 0
#         for idx, coeff in enumerate(self.coefficients[::-1]):
#             ret += coeff*idx*(x**(idx-1))
#         return ret
    
#     def get_min_idx(self):
#         return -self.coefficients[1]/(2*self.coefficients[0])

    

# # Random generating
# def convex_function_generate(n):
#     ret = []
#     for _ in range(n):
#         coeff = np.random.randint(1,10, size=3)
#         ret.append(Polynomial(coeff))
#     return ret

# # Rising generating
# def rising_function_generate(n):
#     ret = []
#     for i in range(n):
#         ret.append(Polynomial([1,-i,0]))
#     return ret

#Random environmetn from a ball
def random_square_loss_generator(d:int):
    vec = np.random.uniform(size=d)
    vec = vec/np.linalg.norm(vec)
    x = np.random.uniform(size=d)
    x = x/np.linalg.norm(x)
    y = np.dot(vec,x)
    y = y +np.random.normal(0,0.01)

    return vec, y

def project(x:np.ndarray, r):
    if np.linalg.norm(x) > r:
        x = x * (r / np.linalg.norm(x))
    return x





class Environment:

    def __init__(self, d=3, T=5000):
        self.d = d
        self.vecs = []
        self.labels = []
        self.T = T

        for i in range(T):
            vec, y = random_square_loss_generator(self.d)
            self.vecs.append(vec)
            self.labels.append(y)

    def get_loss_val(self, t:int, x:np.ndarray):
        assert t < self.T
        return 0.2*(np.dot(self.vecs[t],x)-self.labels[t])**2

    def get_loss_grad(self, t:int,):
        assert t < self.T
        return self.vecs[t]





if __name__ == "__main__":
    s = random_square_loss_generator(3)
