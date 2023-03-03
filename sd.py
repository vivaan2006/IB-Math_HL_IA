#importing a library that stores the value of all the stocks.
import yfinance as yf
# importing a number library that allows for operations to numbers
import numpy as np
# downloads the ticker symbol, and analyzes the stock '$XOM' from specified periods the stock ticker is changed every new stock
prices = yf.download('SHW', start='2016-01-01', end='2023-02-28')['Adj Close']
# the return value variable made, removes the excess digits, and pct_change changes the dataframe and returns the percent difference
returns = prices.pct_change().dropna()
# creates a variable for average return, and np.mean takes the average of whatever is passed in it's parenthesis, in this case the return.
average_return = np.mean(returns)
# the deviations are now calculated by subtracting return and the average return, as can be seen from the formula in the formula section.
deviations = returns - average_return
# the (xi-x0) from the equation 4 is squared, so we can multiply it by itself here.
squared_deviations = deviations ** 2
# the sum of the squared deviations must be taken, and the .sum does that
sum_squared_deviations = np.sum(squared_deviations)
# finally the square root of all those values are taken
std_dev = np.sqrt(sum_squared_deviations / (len(returns) - 1))
# the standard deviation is printed.
print(std_dev)

