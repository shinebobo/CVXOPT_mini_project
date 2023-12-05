import numpy as np
import matplotlib.pyplot as plt
from environment import Polynomial, Environment
from abc import ABC

class Base(ABC):

    def __init__(self, lr: float):
        self.lr = lr
    
    def choose(self, env:Environment, x: float, t: int):
        "Choose next x"
        pass

class OGD(Base):

    def __init__(self, lr: float):
        super().__init__(lr)
    
    def choose(self, env: Environment, x: float, t: int):
        grad = env.get_loss_ft(t).get_grad(x)
        return x - self.lr*grad

class BGD_1(Base):
    def __init__(self, lr: float):
        super().__init__(lr)
    
    def choose(self, env: Environment, x: float, t: int):
        unit_x = np.random.random()
        grad_est = (env.get_loss_val(t,x+unit_x)-env.get_loss_val(t,x-unit_x))/(2*unit_x)
        return x-self.lr*grad_est