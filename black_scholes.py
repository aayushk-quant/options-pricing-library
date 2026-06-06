import numpy as np
from scipy.stats import norm
from option_contract import OptionContract

class BlackScholes:
    def __init__(self, option: OptionContract):
        self.option = option
    def _calculate_d1_d2(self):
        S, K, T, r, sigma = self.option.S, self.option.K, self.option.T, self.option.r, self.option.sigma
        d1 = (np.log(S / K)+ (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    def price(self):
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r = self.option.S, self.option.K, self.option.T, self.option.r
        if self.option.option_type == 'call':
            price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        else:
            price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
        return price
    def delta(self):
        d1 = self._calculate_d1_d2()[0]
        if self.option.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        return delta
    def gamma(self):
        d1 = self._calculate_d1_d2()[0]
        S, T, sigma = self.option.S,  self.option.T, self.option.sigma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    def vega(self):
        d1 = self._calculate_d1_d2()[0]
        S, T = self.option.S, self.option.T
        vega = (S * norm.pdf(d1)) * np.sqrt(T)
        return vega / 100 #per 1% change in volatility
    def theta(self):
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r, sigma = self.option.S, self.option.K, self.option.T, self.option.r, self.option.sigma
        if self.option.option_type == 'call':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        return theta / 365 #daily theta rate
    def rho(self):
        d2 = self._calculate_d1_d2()[1]
        K, T, r = self.option.K, self.option.T, self.option.r
        if self.option.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return rho / 100 #per 1% change in risk-free interest rate
    def IV_solver(self, market_price):
        original_sigma = self.option.sigma
        sigma = self.option.sigma
        tolerance = 1e-6
        max_iter = 1000
        intrinsic = max(self.option.S - self.option.K, 0) if self.option.option_type == 'call' else max (self.option.K - self.option.S, 0) 
        if market_price < intrinsic:
            raise ValueError('Market Price is lower than intrinsic value')
        if self.option.option_type == 'call' and market_price > self.option.S:
            raise ValueError('Call price cannot exceed stock price')
        if self.option.option_type == 'put' and market_price > self.option.K * np.exp(-self.option.r * self.option.T):
            raise ValueError('Put price cannot exceed discounted strike')
        try:
            for _ in range(max_iter):
                self.option.sigma = sigma
                price_diff = self.price() - market_price
                if abs(price_diff) < tolerance:
                    return sigma
                vega = self.vega() * 100
                if abs(vega) < 1e-8:
                    raise ValueError('Vega too small for Newton-Raphson Method')
                sigma -= price_diff / vega 
                if sigma <= 0 or not np.isfinite(sigma):
                    raise ValueError('Invalid sigma during IV solving')
            raise ValueError('IV solver did not converge')        
        finally: 
            self.option.sigma = original_sigma