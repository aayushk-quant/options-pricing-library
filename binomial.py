import numpy as np
from option_contract import OptionContract
import matplotlib.pyplot as plt
from black_scholes import BlackScholes

class Binomial:
    def __init__(self, option: OptionContract, N: int):
        self.option = option
        self.N = N
        if N <= 0:
            raise ValueError("N must be larger than 0")
    def _calculate_parameters(self): #CRR parameterisation used
        T, r, sigma, N = self.option.T, self.option.r, self.option.sigma, self.N
        u = np.exp(sigma * np.sqrt(T / N))
        d = 1 / u
        p = (np.exp(r * T / N) - d) / (u - d)
        if not (0 < p < 1):
            raise ValueError("Risk-neutral probability is not between 0 and 1, check inputs again")
        return u, d, p
    def price(self):
        u, d, p = self._calculate_parameters()
        S, K, T, r, N = self.option.S, self.option.K, self.option.T, self.option.r, self.N
        i = np.arange(N + 1)
        ST = S * (u ** i) * (d ** (N - i))
        if self.option.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        for t in range (N):
            payoffs = np.exp(-r * (T / N)) * (p * payoffs[1:] + (1 - p) * payoffs[:-1])
        return payoffs[0]
    def price_american(self): #Backward induction: at each node take max of continuation value and early exercise
        u, d, p = self._calculate_parameters()
        S, K, T, r, N = self.option.S, self.option.K, self.option.T, self.option.r, self.N
        i = np.arange(N + 1)
        ST = S * (u ** i) * (d ** (N - i))
        if self.option.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        for t in range (N - 1, -1, -1):
            j = np.arange(t + 1)
            S_nodes = S * (u ** j) * (d ** (t - j))
            continuation = np.exp(-r * (T / N)) * (p * payoffs[1:t + 2] + (1 - p) * payoffs[:t + 1])
            if self.option.option_type == "call":
                intrinsic = np.maximum(S_nodes - K, 0)
            else:
                intrinsic = np.maximum(K - S_nodes, 0)
            payoffs = np.maximum(continuation, intrinsic)
        return payoffs[0]
    def plot_convergence(self):
        N_values = range(10, 500, 10)
        prices = [Binomial(self.option, N).price() for N in N_values]
        bs_price = BlackScholes(self.option).price()
        plt.plot(N_values, prices)
        plt.xlabel("Number of Steps")
        plt.ylabel("Price over time")
        plt.title("Binomial Stock Price per Step")
        plt.axhline(y = bs_price, color = 'r', linestyle = '--', label = 'Black-Scholes')
        plt.legend()
        plt.show()  