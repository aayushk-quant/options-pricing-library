import numpy as np
from option_contract import OptionContract
from black_scholes import BlackScholes 
import matplotlib.pyplot as plt
import yfinance as yf

class VolSmile:
    def __init__(self, option: OptionContract, strike_prices: np.ndarray, market_prices: np.ndarray):
        self.option = option
        self.strike_prices = strike_prices
        self.market_prices = market_prices
    def calculate_ivs(self):
        ivs = []
        for K, market_price in zip(self.strike_prices, self.market_prices):
            S, T, r, sigma, option_type = self.option.S, self.option.T, self.option.r, self.option.sigma, self.option.option_type
            contract = OptionContract(S, K, T, r, sigma, option_type)
            bs = BlackScholes(contract)
            iv = bs.IV_solver(market_price)
            ivs.append(iv)
        return np.array(ivs)
    def plot(self):
        ivs = self.calculate_ivs()
        plt.plot(self.strike_prices, ivs)
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title('Volatility Smile')
        plt.show()
    def plot_market_smile(self, ticker: str, expiry: str):
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        calls = chain.calls
        spot = tk.info['currentPrice']
        calls = calls[(calls['strike'] > 0.7 * spot) & (calls['strike'] < 1.3 * spot)]
        plt.plot(calls['strike'], calls['impliedVolatility'])
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title('Volatility Smile')
        plt.show()