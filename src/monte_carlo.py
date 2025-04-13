import numpy as np
import matplotlib.pyplot as plt


def monte_carlo(S: float, T: float, r: float, q: float, sigma: float, steps: int, N: int) -> np.array:
    """
    Inputs
    S - Current stock Price

    T - Time to maturity 1 year = 1, 1 months = 1/12

    K - Strike Price

    r - risk free interest rate

    q - dividend yield

    sigma - volatility

    steps - number of steps in integration
    
    N - number of Monte-Carlo paths 
    
    Output
    Matrix of asset paths [steps,N]  
    """
    dt = T/steps
    #S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)