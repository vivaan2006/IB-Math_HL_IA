import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

tickers = ['XOM', 'SHW', 'BA', 'UNH', 'AMZN']
startDate = '2016-01-01'
endDate = '2023-02-28'
prices = yf.download(tickers, start=startDate, end=endDate)['Adj Close']

# Compute the covariance matrix, and add find returns.
returns = prices.pct_change().dropna()
cov = returns.cov()

# Define the objective function
def portfolioVariance(weights, covariance):
    return np.dot(weights.T, np.dot(covariance, weights))

# Define the constraintst
def constraints(weights):
    return np.sum(weights) - 1
# Set the initial guess, and bounds
InitialGuess = np.ones(returns.shape[1]) / returns.shape[1]
bounds = [(0, 1) for i in range(returns.shape[1])]

# Minimize the function depending on the constraints provided. Next line retrieves the weights.
result = minimize(portfolioVariance, InitialGuess, args=(cov,), method='SLSQP', constraints={'type': 'eq', 'fun': constraints}, bounds=bounds)
optimalWeights = result.x

# Make the optimal portfolio
returnOfPortfolio = np.dot(optimalWeights.T, returns.mean())
volatility = np.sqrt(portfolioVariance(optimalWeights, cov))
sharpeRatio = returnOfPortfolio / volatility

print('Optimal portfolio weights: ', optimalWeights)
print('Portfolio return: ', returnOfPortfolio)
print('Portfolio volatility: ', volatility)
print('Sharpe ratio: ', sharpeRatio)

# The efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined level
# of risk Or the lowest risk for a given level of expected return. This computes that.
target_returns = np.linspace(returns.mean().min(), returns.mean().max(), 100)
efficientFrontier = []

for target_return in target_returns:
    def ObjectiveFunction(weights):
        return -np.dot(weights.T, returns.mean()) + target_return * np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    result = minimize(ObjectiveFunction, InitialGuess, method='SLSQP', constraints={'type': 'eq', 'fun': constraints}, bounds=bounds)
    efficientFrontier.append((target_return, np.sqrt(portfolioVariance(result.x, cov))))
efficientFrontier = np.array(efficientFrontier)

plt.plot(efficientFrontier[:, 1], efficientFrontier[:, 0], label='Efficient frontier')
plt.scatter(volatility, returnOfPortfolio, color='red', label='Optimal portfolio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.show()
