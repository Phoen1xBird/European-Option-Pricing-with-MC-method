import numpy as np
from scipy.stats import norm

def black_scholes_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Inputs
    S - Current stock Price

    K - Strike Price

    T - Time to maturity 1 year = 1, 1 months = 1/12

    r - risk free interest rate

    q - dividend yield

    sigma - volatility 
    
    Output
    # call_price = value of the option 
    """
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    
    call = S * np.exp(-q*T)* norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
    return call