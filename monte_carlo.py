import numpy as np
from option_contract import OptionContract
import matplotlib.pyplot as plt
from black_scholes import BlackScholes

class MonteCarlo:
    def __init__(self, option: OptionContract, n_steps: int = 252, n_simulations: int = 10000, seed: int = None):
        self.option = option
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.seed = seed
        if n_simulations % 2 != 0: 
            raise ValueError('n_simulations must be even when using antithetic variates')
    def _simulate_paths(self): #Assumed geometric brownian motion under risk-neutral measure
        n_simulations, n_steps = self.n_simulations, self.n_steps
        S, T, r, sigma = self.option.S, self.option.T, self.option.r, self.option.sigma
        if self.seed is not None:
            np.random.seed(self.seed)
        Z = np.random.standard_normal((n_simulations // 2, n_steps)) #Antithetic variates for variance reduction
        Z = np.vstack([Z, -Z])
        dt = T / n_steps
        increments = (r - (sigma ** 2) * 0.5) * dt + (sigma * np.sqrt(dt) * Z)
        S0 = np.full((n_simulations, 1), S)
        paths = np.hstack([S0, S * np.exp(np.cumsum(increments, axis=1))])
        return paths
    def price(self):
        paths, r, T, K = self._simulate_paths(), self.option.r, self.option.T, self.option.K
        ST = paths[:, -1]
        if self.option.option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
    def price_asian(self):
        paths, r, T, K = self._simulate_paths(), self.option.r, self.option.T, self.option.K
        S_avg = np.mean(paths, axis = 1)
        if self.option.option_type == 'call':
            payoffs = np.maximum(S_avg - K, 0)
        else:
            payoffs = np.maximum(K - S_avg, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
    def price_barrier(self, barrier: float, barrier_type: str): #barrier type can be 'd_o', 'd_i', 'u_o', 'u_i'
        paths, r, T, K = self._simulate_paths(), self.option.r, self.option.T, self.option.K
        ST = paths[:, -1]
        S_min = np.min(paths, axis = 1)
        S_max = np.max(paths, axis = 1)
        if self.option.option_type == 'call':
            vanilla_payoff = np.maximum(ST - K, 0)
        else:
            vanilla_payoff = np.maximum(K - ST, 0)
        if barrier_type == 'd_o':
            payoffs = np.where(S_min > barrier, vanilla_payoff, 0)
        elif barrier_type == 'd_i':
            payoffs = np.where(S_min > barrier, 0, vanilla_payoff)
        elif barrier_type == 'u_o':
            payoffs = np.where(S_max < barrier, vanilla_payoff, 0)
        elif barrier_type == 'u_i':
            payoffs = np.where(S_max < barrier, 0, vanilla_payoff)
        else:
            raise ValueError("barrier_type must be 'd_o', 'd_i', 'u_o', or 'u_i'")
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
    def price_with_ci(self):
        paths, r, T, K = self._simulate_paths(), self.option.r, self.option.T, self.option.K
        ST = paths[:, -1]
        if self.option.option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        discounted_payoffs = np.exp(-r * T) * payoffs
        price = np.mean(discounted_payoffs)
        se = np.std(discounted_payoffs, ddof = 1) / np.sqrt(self.n_simulations)
        ci = [price - 1.96 * se, price + 1.96 * se]
        return price, ci
    def plot_convergence(self):
        N_values = range(10, 10000, 10)
        prices = [MonteCarlo(self.option, n_simulations=N).price() for N in N_values]
        bs_price = BlackScholes(self.option).price()
        plt.plot(N_values, prices)
        plt.xlabel("Number of Simulations")
        plt.ylabel("Price over time")
        plt.title("MonteCarlo Convergence")
        plt.axhline(y = bs_price, color = 'r', linestyle = '--', label = 'Black-Scholes')
        plt.legend()
        plt.show()  