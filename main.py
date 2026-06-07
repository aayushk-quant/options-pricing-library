from option_contract import OptionContract
from black_scholes import BlackScholes
from binomial import Binomial
from monte_carlo import MonteCarlo
from vol_smile import VolSmile
import numpy as np

if __name__ == "__main__":
    Testc = OptionContract(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    Testp = OptionContract(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
    resultcbs = BlackScholes(Testc)
    resultpbs = BlackScholes(Testp)
    resultbc = Binomial(Testc, 1000)
    resultbp = Binomial(Testp, 1000)
    resultmcc = MonteCarlo(Testc)
    resultmcp = MonteCarlo(Testp)
    strikes = np.linspace(80, 120, 20)
    market_prices = [BlackScholes(OptionContract(100, i, 1, 0.05, 0.2 + 0.1 * ((i - 100) / 100) ** 2, 'call')).price() for i in strikes]
    pricec, cic = resultmcc.price_with_ci()
    pricep, cip = resultmcp.price_with_ci()

    print('\n===Black Scholes===')
    print(f'Call price: {resultcbs.price():.4f}')
    print(f'Put price: {resultpbs.price():.4f}')
    print('\n===Greeks===')
    print(f'Call: \nDelta: {resultcbs.delta():.4f}, Gamma: {resultcbs.gamma():.4f}, Vega = {resultcbs.vega():.4f}, Theta = {resultcbs.theta():.4f}, Rho = {resultcbs.rho():.4f}')
    print(f'Put: \nDelta: {resultpbs.delta():.4f}, Gamma: {resultpbs.gamma():.4f}, Vega = {resultpbs.vega():.4f}, Theta = {resultpbs.theta():.4f}, Rho = {resultpbs.rho():.4f}')
    print('\n===Binomial===')
    print(f'European Call price: {resultbc.price():.4f}')
    print(f'European Put price: {resultbp.price():.4f}')
    print(f'American Call price: {resultbc.price_american():.4f}') #Added for completeness, in reality never optimal to exit early (assuming no dividends)
    print(f'American Put price: {resultbp.price_american():.4f}')
    print('\n===Monte Carlo===')
    print(f'European Call price: {resultmcc.price():.4f}')
    print(f'European Put price: {resultmcp.price():.4f}')
    print(f'European Call 95% CI: [{cic[0]:.4f}, {cic[1]:.4f}]')
    print(f'European Put 95% CI: [{cip[0]:.4f}, {cip[1]:.4f}]')
    print(f'Asian Call price: {resultmcc.price_asian():.4f}')
    print(f'Asian Put price: {resultmcp.price_asian():.4f}')
    print(f'Barrier Call price: {resultmcc.price_barrier(90, "d_o"):.4f}')
    print(f'Barrier Put price: {resultmcp.price_barrier(90, "d_o"):.4f}')
    print('\n===IV Solver===')
    print(f'Implied Volatility: {resultcbs.iv_solver(resultcbs.price()):.4f}')
    print('\n===Volatility Smile===')
    print('Generating volatility smile plots: one synthetic, one from real AAPL market data...')
    VolSmile(Testc, strikes, market_prices).plot()
    VolSmile(Testc, strikes, market_prices).plot_market_smile('AAPL')
    print('\n===Convergence Plots===')
    print('Generating convergence plots for Binomial and Monte Carlo...')
    resultbc.plot_convergence()
    resultmcc.plot_convergence()