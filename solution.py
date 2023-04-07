import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm

chat_id = 894208417 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array) -> float:
    def neg_log_likelihood(a, x):
        return -lognorm.logpdf(x, s=1, scale=np.exp(a)).sum()

    res = minimize_scalar(neg_log_likelihood, args=(x,), method='bounded', bounds=(0, None))

    return res.x
    #return x.mean() # Ваш ответ
