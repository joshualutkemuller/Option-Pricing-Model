import math
import random 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Black-Scholes model for valuing a swaption
def black_scholes_swaption(principal, strike, volatility, expiration, risk_free_rate, option_type):
    d1 = (math.log(principal / strike) + (risk_free_rate + 0.5 * volatility**2) * expiration) / (volatility * math.sqrt(expiration))
    d2 = d1 - volatility * math.sqrt(expiration)
    
    if option_type == 'Call':
        return principal * (norm_cdf(d1) - strike * math.exp(-risk_free_rate * expiration) * norm_cdf(d2))
    elif option_type == 'Put':
        return principal * (strike * math.exp(-risk_free_rate * expiration) * norm_cdf(-d2) - norm_cdf(-d1))
    else:
        raise ValueError("Invalid option type")

# Hull-White model for valuing a swaption
def hull_white_swaption(principal, strike, volatility, expiration, risk_free_rate, mean_reversion, option_type):
    a = (1 - math.exp(-mean_reversion * expiration)) / mean_reversion
    d1 = (math.log(principal / strike) + 0.5 * (volatility**2 / mean_reversion**2) * (1 - math.exp(-2 * mean_reversion * expiration))) / (volatility / mean_reversion * math.sqrt(1 - math.exp(-2 * mean_reversion * expiration)))
    d2 = d1 - volatility / mean_reversion * math.sqrt(1 - math.exp(-2 * mean_reversion * expiration))
    
    if option_type == 'Call':
        return principal * (norm_cdf(d1) - strike * math.exp(-risk_free_rate * expiration) * norm_cdf(d2))
    elif option_type == 'Put':
        return principal * (strike * math.exp(-risk_free_rate * expiration) * norm_cdf(-d2) - norm_cdf(-d1))
    else:
        raise ValueError("Invalid option type")

# Heston model for valuing a swaption
def heston_swaption(principal, strike, expiration, risk_free_rate, kappa, theta, sigma, rho, v0, option_type, num_steps=1000, num_paths=1000):
    dt = expiration / num_steps
    sqrt_dt = math.sqrt(dt)
    num_iterations = int(expiration * num_steps)
    
    option_value = 0
    
    for _ in range(num_paths):
        vt = v0
        st = principal
        
        for _ in range(num_iterations):
            z1 = random.gauss(0, 1)
            z2 = random.gauss(0, 1)
            
            dz1 = z1 * sqrt_dt
            dz2 = rho * z1 * sqrt_dt + math.sqrt(1 - rho**2) * z2 * sqrt_dt
            
            vt = max(vt + kappa * (theta - vt) * dt + sigma * math.sqrt(vt) * dz2, 0)
            st *= math.exp((risk_free_rate - 0.5 * vt) * dt + math.sqrt(vt) * dz1)
        
        if option_type == 'Call':
            option_value += max(st - strike, 0)
        elif option_type == 'Put':
            option_value += max(strike - st, 0)
        else:
            raise ValueError("Invalid option type")
    
    option_value /= num_paths
    option_value *= math.exp(-risk_free_rate * expiration)
    
    return option_value

# Helper function for calculating the cumulative distribution function (CDF) of the standard normal distribution
def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

# Example usage
principal = 10000000  # Principal amount
strike = 0.05  # Strike rate
volatility = 0.4  # Volatility
expiration = 1.0  # Time to expiration (in years)
risk_free_rate = 0.05  # Risk-free interest rate
mean_reversion = 3  # Mean reversion parameter (for Hull-White model)
kappa = 2.0  # Mean reversion parameter (for Heston model)
theta = 0.05  # Long-run variance (for Heston model)
sigma = 0.2  # Volatility of volatility (for Heston model)
rho = -0.5  # Correlation between the asset price and volatility (for Heston model)
v0 = 0.05  # Initial variance (for Heston model)

option_type = 'Call'  # Option type: 'Call' or 'Put'

# Valuation using Black-Scholes model
bs_value = black_scholes_swaption(principal, strike, volatility, expiration, risk_free_rate, option_type)
print("Black-Scholes value:", bs_value)

# Valuation using Hull-White model
hw_value = hull_white_swaption(principal, strike, volatility, expiration, risk_free_rate, mean_reversion, option_type)
print("Hull-White value:", hw_value)

# Valuation using Heston model
heston_value = heston_swaption(principal, strike, expiration, risk_free_rate, kappa, theta, sigma, rho, v0, option_type)
print("Heston value:", heston_value)

#Set the style of seaborn for plots of different pricing models
sns.set(style='whitegrid')

df = pd.DataFrame({"Model" : ['Black Scholes','Hull-White','Heston'],
              "Model Outputs (Valuation $)": [bs_value, hw_value, heston_value]})

plt.figure(figsize = (20,5))
sns.barplot(x ='Model',y='Model Outputs (Valuation $)',data = df)
plt.title('Distribution of Option Model Valuations')
plt.show()
