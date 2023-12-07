import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, project
from abc import ABC

class Base(ABC):

    def __init__(self, lr: float):
        self.lr = lr
    
    def choose(self, env:Environment, x: np.ndarray, t: int):
        "Choose next x"
        pass

class OGD(Base):

    def __init__(self, lr: float):
        super().__init__(lr)
        
    
    def choose(self, env: Environment, x: np.ndarray, t: int):
        grad = env.get_loss_grad(t-1)
        return x - self.lr*grad

class BGD_1(Base):
    def __init__(self, lr: float, pr:float, d:int):
        super().__init__(lr)
        self.pr = pr
        self.d = d
        self.y = np.zeros(d)
        
    def choose(self, env: Environment, t: int):
        unit_x = np.random.normal(size=env.d)
        unit_x = unit_x/np.linalg.norm(unit_x)
        ret = self.y + unit_x*self.pr
        loss = env.get_loss_val(t,self.y + unit_x*self.pr)
        grad_est = env.d/(self.pr)*loss*unit_x
        self.y = self.y-self.lr*grad_est
        self.y = project(self.y, 1-self.pr)
        return ret


class BGD_2(Base):
    def __init__(self, lr: float, pr:float, d=int):
        super().__init__(lr)
        self.pr = pr
        self.y = np.zeros(d)

    
    def choose(self, env: Environment, t: int):
        unit_x = np.random.normal(size=env.d)
        unit_x = unit_x/np.linalg.norm(unit_x)
        ret1 = self.y + unit_x*self.pr
        ret2 = self.y - unit_x*self.pr
        loss1 = env.get_loss_val(t,self.y + unit_x*self.pr)
        loss2 = env.get_loss_val(t,self.y - unit_x*self.pr)

        grad_est = env.d/(2*self.pr)*(loss1-loss2)*unit_x
        self.y = self.y-self.lr*grad_est
        self.y = project(self.y, 1-self.pr)
        return ret1, ret2

class BGD_k(Base):
    def __init__(self, lr: float, pr:float, d=int, k=int):
        super().__init__(lr)
        self.pr = pr
        self.y = np.zeros(d)
        self.k = k

    
    def choose(self, env: Environment, t: int):
        unit_x = np.random.normal(size=env.d)
        unit_x = unit_x/np.linalg.norm(unit_x)
        delta = np.log(t+1)/(t+1)
        ret = [self.y]
        coords = np.eye(self.k)
        grad_est = np.zeros(self.k)
        
        for i in range(self.k):
            ret.append(self.y+delta*coords[i])
            grad_est += (env.get_loss_val(t,self.y)-env.get_loss_val(t,self.y+delta*coords[i]))*coords[i]

        grad_est = grad_est/delta
        self.y = self.y-self.lr*grad_est
        self.y = project(self.y, 1-self.pr)
        return ret