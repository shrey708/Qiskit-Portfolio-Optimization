from data_load import Data_load
from portfolio_bitstring import portfolio_opt
from weights_allocation import opt_weights, get_portfolio_sharpe
import datetime as datetime 
# from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import pandas as pd


class run_app:
    
    def __init__(self, index, assets, budget, past_years, algorithm, device, trading_days) -> None:
        self.index = index
        self.assets = assets 
        self.budget = budget
        self.past_years = past_years
        self.algorithm = algorithm
        self.device = device 
        self.trading_days = trading_days
    
    
    
    def app(self):
            
        end_date = datetime.date.today()
        if self.past_years == "5 years":
            past_years = 5
            
        elif self.past_years == "10 years":
            past_years = 10
        
        start_date  = datetime.date.today() - relativedelta(years=past_years)
        data = Data_load(self.assets, start_date, end_date)
        log_returns= data.get_data()

        print("returns are ", log_returns)
        opt_bit = portfolio_opt(self.assets, log_returns, self.budget, self.device, trading_days = 252)
        if self.algorithm ==  "QAOA with cobyla":
            opt_bitstring  = opt_bit.get_solution_using_qaoa_cobyla()
        elif self.algorithm ==  "QAOA with SPSA":
            opt_bitstring  = opt_bit.get_solution_using_qaoa_spsa()
        elif self.algorithm ==  "VQE with cobyla":
            opt_bitstring  = opt_bit.get_solution_using_vqe_cobyla()
        
        elif self.algorithm ==  "VQE with SPSA":
            opt_bitstring  = opt_bit.get_solution_using_vqe_spsa()
        
        
        
        # opt_bitstring 
        
        my_weight  = opt_weights(log_returns, opt_bitstring)
        my_weights = my_weight.optimize_weights()
        # my_weights
        
        sharpe = get_portfolio_sharpe(log_returns, my_weights)
        sharpe_ratio = sharpe.get_sharpe_ratio()
        
        return my_weights, sharpe_ratio
        
