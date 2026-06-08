import numpy as np
from option_contract import OptionContract
from black_scholes import BlackScholes 
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

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
            iv = bs.iv_solver(market_price)
            ivs.append(iv)
        return np.array(ivs)
    def plot(self):
        ivs = self.calculate_ivs()
        plt.plot(self.strike_prices, ivs)
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title('Volatility Smile')
        plt.show()
    def plot_market_smile(self, ticker: str, expiry: str = None):
        tk = yf.Ticker(ticker)
        options = tk.options
        if not options:
            raise ValueError(f'No option expiries found for {ticker}')
        if expiry is None:
            expiry = options[min(1, len(options) - 1)]
        if expiry not in options:
            raise ValueError(f"Expiry {expiry} not available for {ticker}")
        chain = tk.option_chain(expiry)
        calls = chain.calls
        spot = tk.info.get('currentPrice')
        if spot is None or spot <= 0:
            raise ValueError(f"Could not retrieve valid spot price for {ticker}")
        T = (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.today().date()).days / 365
        if T <= 0:
            raise ValueError("Expiry must be in the future")
        calls = calls[
            (calls['strike'] > 0.7 * spot) &
            (calls['strike'] < 1.3 * spot) &
            (calls['bid'] > 0) &
            (calls['ask'] > 0)
        ].copy()
        if calls.empty:
            raise ValueError("No valid option contracts after filtering")
        mid_prices = (calls['bid'] + calls['ask']) / 2
        r = self.option.r
        sigma_init = self.option.sigma
        ivs = []
        for K, market_price in zip(calls['strike'], mid_prices):
            try:
                contract = OptionContract(spot, K, T, r, sigma_init, 'call')
                iv = BlackScholes(contract).iv_solver(market_price)
            except (ValueError, RuntimeError):
                iv = np.nan
            ivs.append(iv)
        strikes = calls['strike'].to_numpy()
        ivs = np.array(ivs)
        mask = ~np.isnan(ivs)
        if not np.any(mask):
            raise ValueError("No valid implied volatilities could be calculated")
        plt.plot(strikes[mask], ivs[mask])
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title(f'{ticker} Volatility Smile ({expiry})')
        plt.show()