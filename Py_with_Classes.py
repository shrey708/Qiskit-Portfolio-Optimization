#importing the required stuff
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
import csv
import yfinance as yf
import pandas_datareader.data as web
from qiskit_finance.applications.optimization import PortfolioDiversification
#%matplotlib inline



#Define assets and store them globally
assets = []
# number of elements as input
n = int(input("Enter number of elements : "))
  
# iterating till the range
for i in range(0, n):
    ele = str(input())
    assets.append(ele) # adding the element 
print(assets)

num_assets = len(assets)


class download_plots():
    def values(self, mu=float, sigma=float):
    # set number of assets (= number of qubits)
    # Using data from yahoo finance
    # use the name listed for particular company that you wish to add
        self.seed = 123
        selected_stocks_data = yf.download(assets, start = '2013-01-01', end = '2021-12-31')['Adj Close']
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


#class risk_fact:
    def index_to_selection(self):
        s = "{0:b}".format(i).rjust(num_assets)
        x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
        

        q = 0.5  # set risk factor
        budget = num_assets // 2  # set budget
        self.penalty = num_assets  # set parameter to scale the budget penalty term
        portfolio = PortfolioOptimization(expected_returns=download_plots.mu, covariances=download_plots.sigma, risk_factor=q, budget=budget,)
        qp = portfolio.to_quadratic_program()
        qp

        return x
    
#class print_selection():   
    def print_result(self,result,fname):
        f = open(fname, 'w')
        header = ['SELECTION', 'VALUE', 'PROBABLITY']
        writer = csv.writer(f)
        writer.writerow(header)
    
        selection = result.x
        value = result.fval
        print("Optimal: selection {}, value {:.4f}".format(selection, value))




