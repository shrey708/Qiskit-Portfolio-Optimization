#importing the required stuff
from unittest import result
from qiskit import Aer
from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import *
from qiskit_finance import QiskitFinanceError
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from qiskit.utils import algorithm_globals
from scipy.optimize import minimize
import yfinance as yf
import pandas_datareader.data as web
from qiskit_finance.applications.optimization import PortfolioDiversification
import datetime
#%matplotlib inline



def __init__(self, assets, date1, date2): #Define assets and store them globally
    self.assets=assets
    self.date1=date1
    self.date2=date2

    assets = []
    # number of elements as input
    n = int(input("Enter number of elements : "))
  
    # iterating till the range
    for i in range(0, n):
        ele = str(input())
        self.append(ele) # adding the element
    print('Your selection:',assets)
    self.num_assets = len(assets)


    self.date_entry1 = input('Enter a date in YYYY-MM-DD format')
    year, month, day = map(int, self.split('-'))
    date1 = datetime.date(year, month, day)
    print('starting',date1)

    # loadinf end date
    self.date_entry2 = input('Enter a date in YYYY-MM-DD format')
    year, month, day = map(int, self.split('-'))
    date2 = datetime.date(year, month, day)
    print('starting',date2)
    return assets,date1, date2


class loading_plots:
    def values(self, mu, sigma):
    # set number of assets (= number of qubits)
    # Using data from yahoo finance
    # use the name listed for particular company that you wish to add
        self.seed = 123
        selected_stocks_data = yf.download(__init__.assets, start = '2013-01-01', end = '2021-12-31')['Adj Close']
        selected_stocks_data.head()
        self.log_return = np.log(selected_stocks_data/selected_stocks_data.shift(1))
    #np.log(data/data.shift(1))
    #2log_return.head()
        
        self.mu = self.log_return.mean()*252
        self.sigma.sigma = self.log_return.cov()*252
        self.sigma.sigma = np.array(sigma)
        self.mu = np.array(mu)

        print("returns ", mu)
        print("cov ", sigma)
        # plot sigma
        plt.imshow(sigma, interpolation="nearest")
        plt.show()


#risk_fact:
    def index_to_selection(self,qp):
        s = "{0:b}".format(self.i).rjust(__init__.num_assets)
        x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(__init__.num_assets))])
        

        q = 0.5  # set risk factor
        budget = __init__.num_assets // 2  # set budget
        self.penalty = __init__.num_assets  # set parameter to scale the budget penalty term
        portfolio = PortfolioOptimization(expected_returns=self.download_plots.mu, covariances=self.download_plots.sigma, risk_factor=q, budget=budget,)
        qp = portfolio.to_quadratic_program()
        qp

        return x,qp
    
#print_selection():   
    def print_result(self,fname):
        f = open(fname, 'w')
        selection = self.result.x
        value = self.result.fval
        print("Optimal: selection {}, value {:.4f}".format(selection, value))
        return selection, value


class classical(result):
    #classical reference solution
    #provide fname ='complete address/location of your .csv file'

    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)

    result = exact_eigensolver.solve(loading_plots.index_to_selection.qp)
    fname = 'nes1.csv' #change the location and nsame of .csv file

    loading_plots.print_result(result,fname)

class vqe(result):
    algorithm_globals.random_seed = 1234
    backend = Aer.get_backend("qasm_simulator")

    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    ry = TwoLocal(__init__.num_assets, "ry", "cz", reps=3, entanglement="full")
    quantum_instance = QuantumInstance(backend=backend, shots=8192, seed_simulator=loading_plots.values.seed, seed_transpiler=loading_plots.values.seed)
    vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
    vqe = MinimumEigenOptimizer(vqe_mes)
    result = vqe.solve(loading_plots.index_to_selection.qp)

    fname = 'vqe1.csv' #change the location and name of .csv file
    loading_plots.print_result(result, fname)

class qaoa(result):

    def qaoa_sol(self,fname):
        # QAOA solution
        #provide fname ='complete address/location of your .csv file'

        algorithm_globals.random_seed = 1234
        backend = Aer.get_backend("statevector_simulator")

        cobyla = COBYLA()
        cobyla.set_options(maxiter=250)
        quantum_instance = QuantumInstance(backend=backend, shots=8192, seed_simulator=loading_plots.values.seed, seed_transpiler=loading_plots.values.seed)
        qaoa_mes = QAOA(optimizer=cobyla, reps=3, quantum_instance=quantum_instance)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = qaoa.solve(loading_plots.index_to_selection.qp)

        fname = 'qaoa1.csv' #change the location and name of .csv file
        loading_plots.print_result(result, fname)


    def spsa(self,fname):
        # QAOA wth SPSA solution
        #provide fname ='complete address/location of your .csv file'

        algorithm_globals.random_seed = 1234
        backend = Aer.get_backend("statevector_simulator")

        cobyla = SPSA()
        cobyla.set_options(maxiter=250)
        quantum_instance = QuantumInstance(backend=backend, shots=8192, seed_simulator=loading_plots.values.seed, seed_transpiler=loading_plots.values.seed)
        qaoa_mes = QAOA(optimizer=cobyla, reps=3, quantum_instance=quantum_instance)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = qaoa.solve(loading_plots.index_to_selection.qp)
        fname = 'qaoa1_1.csv' #change the location and name of .csv file
        loading_plots.print_result(result, fname)

    def lucky_assets(self, my_assets):
        self.selected_assets = [0, 0, 1, 1, 0, 1]
        self.assets = __init__.assets

        selected_bitstring = [i for i, e in enumerate(self.selected_assets) if e == 1]
        # print(selected_bitstring)
        self.my_assets = [__init__.assets[i] for i in selected_bitstring]
        print("your lucky assets are ",my_assets)

        def lucky_data(self):
            self.selected_stocks_data = yf.download(my_assets, start = __init__.date1, end = __init__.date2)['Adj Close']
            self.selected_stocks_data.head()


class weights:
    def normalize(weights):
        weights = np.array(np.random.random(len(qaoa.lucky_assets.my_assets)))
        print('normalised weights :')
        weights = weights/np.sum(weights)
        print(weights)
        return weights

    def get_ret_vol_sr(weights): 
        weights = np.array(weights)
        ret = np.sum(loading_plots.values.log_return.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T,np.dot(loading_plots.values.log_return.cov()*252,weights)))
        sr = ret/vol 
        return np.array(sr)
    

    # minimize negative Sharpe Ratio
    def neg_sharpe(weights): 
        return weights.get_ret_vol_sr(weights)*-1


    # check allocation sums to 1
    def check_sum(weights): 
        return np.sum(weights) - 1

    
    def constraints(self, cons, bounds):
        # create constraint variable
        init_guess = [0.3, 0.5, 0.2]
        cons = ({'type':'eq','fun':weights.check_sum})

        # create weight boundaries
        self.bounds = tuple((0, 1) for stocks in range(len(qaoa.lucky_assets.my_assets)))

        print(weights.neg_sharpe(weights))

        opt_results = minimize(weights.neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        print(opt_results)

        return cons


    def weight_allo(self, my_weights, tot_weights, weights_alloc):
        my_weights = np.array(weights.opt_results.x)
        my_weights = my_weights.tolist()

        tot_weights = np.sum(my_weights)
        tot_weights

        weights_alloc = dict(zip(qaoa.lucky_assets.my_assets, my_weights))
        weights_alloc


    def sharpe_ratio(self, ret, vol, sharpe_ratio):
        ret = np.sum(loading_plots.values.log_return.mean() * weights.weight_allo.my_weights) * 252 # returns of a portfolio after optimum weight allocation
        vol = np.sqrt(np.dot(weights.T,np.dot(loading_plots.values.log_return.cov()*252,weights.weight_allo.my_weights))) # risk of a portfolio after optimum weight allocation
        sharpe_ratio = ret/vol # sharpe ratio of a portfolio after optimum weight allocation
        print("sharpe ratio of your porrtfolio after optimization is ", sharpe_ratio)
        
        self. risk_ret_dict_of_quantum = {
        'returns' : ret*100,
        'risk' : vol*100, 
        'sharpe_ratio' : sharpe_ratio }  

        self.risk_ret_dict_of_quantum


    def pie_chart(self, y, mylabels):
        y = np.array(weights)
        mylabels = qaoa.lucky_assets.my_assets

        plt.pie(y, labels = mylabels)
        plt.show()
    

class mpt:
    def download_data(self):
        f_date = __init__.date1
        l_date = __init__.date2
        self.delta = l_date - f_date
        self.NUM_TRADING_DAYS = self.delta.days #252*5 cosnidered previously, now I'm just taing for 1 year
        self.NUM_PORTFOLIOS = 10000
        self.stocks = __init__.assets
        stocks_data = yf.download(self.stocks, start = __init__.date1, end = __init__.date2)['Adj Close']
        return pd.DataFrame(stocks_data)

    
    def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
        plt.figure(figsize=(8, 5))
        plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(mpt.statistics(opt['x'], rets)[1], mpt.statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
        plt.show()

    def calculate_return(data):
        log_return = np.log(data / data.shift(1))
        return log_return[1:]

    def show_statistics(returns):
        print(returns.mean() * mpt.download_data.NUM_TRADING_DAYS)
        print(returns.cov() * mpt.download_data.NUM_TRADING_DAYS)

    def statistics(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * mpt.download_dataNUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * mpt.download_dataNUM_TRADING_DAYS, weights)))
        return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

    def show_mean_variance(returns, weights):
        portfolio_return = np.sum(returns.mean() * weights) * mpt.download_data.NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * mpt.download_data.NUM_TRADING_DAYS, weights)))
        print('Expected portfolio mean: ', portfolio_return)
        print('Expected portfolio volatility: ', portfolio_volatility)
        print("sharpe ratio of a portfolio is ", portfolio_return/portfolio_volatility)

    def show_portfolios(returns, volatilities):
        plt.figure(figsize=(8, 5))
        plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

    def generate_portfolios(returns):
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []

        for _ in range(mpt.download_data.NUM_PORTFOLIOS):
            w = np.random.random(len(mpt.download_data.stocks))
            w /= np.sum(w)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(returns.mean() * w) * mpt.download_data.NUM_TRADING_DAYS)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * mpt.download_data.NUM_TRADING_DAYS, w))))
        return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

    def show_data(data):
        data.plot(figsize=(8, 5))
        plt.show()

    def mpt_graph(self):
        if __name__ == '__main__':
            dataset = mpt.download_data()
            mpt.show_data(dataset)
            self.log_daily_returns = mpt.calculate_return(dataset)
            pweights, means, risks = mpt.generate_portfolios(self.log_daily_returns)
            mpt.show_portfolios(means, risks)
            optimum = mpt.optimize_portfolio(pweights, self.log_daily_returns)
            mpt.print_optimal_portfolio(optimum, self.log_daily_returns)
            mpt.show_optimal_portfolio(optimum, self.log_daily_returns, means, risks)
    

    






    