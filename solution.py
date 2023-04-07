import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm

chat_id = 894208417 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array) -> float:
    def log_likelihood(a, x):
        n = len(x)
        return -n*np.log(np.sqrt(2*np.pi)) - n*np.log(a) - np.sum(np.log(x**2)) - np.sum((np.log(x)-np.log(a))**2)/(2*a**2)
    
    def neg_log_likelihood(a, x):
        return -log_likelihood(a, x)
    
    res = minimize_scalar(neg_log_likelihood, args=(x,), method='bounded', bounds=(1e-6, 1e6))
    return res.x
    #return x.mean() # Ваш ответ
