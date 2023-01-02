## Import section
import numpy as np
from scipy.optimize import minimize


class opt_weights:
    def __init__(self, assets, log_returns,opt_bitstring):
        self.log_returns = log_returns
        self.opt_bitstring = opt_bitstring
        # self.returns = returns
        self.assets = assets
    
    
    
    # def get_selected_assets(self):
    def select(self):
        selected_bitstring = [i for i, e in enumerate(self.opt_bitstring) if e == 1]
        # print(selected_bitstring)
        my_assets = [self.assets[i] for i in selected_bitstring]
        return my_assets
        
    


    
    def weights_(self):
        weights = np.array(np.random.random(len(self.select())))

        print('normalised weights :')
        weights = weights/np.sum(weights)
        return weights
        
    def get_ret_vol_sr(self,weights,log_return): 
        weights = np.array(weights)
        ret = np.sum(log_return.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252,weights)))
        sr = ret/vol 
        return np.array(sr)

    def neg_sharpe(self,weights): 
        return self.get_ret_vol_sr(weights,self.log_returns)*-1

    def check_sum(self,weights): 
        return np.sum(weights) - 1
    
    def round_w(self):
        round_weights = []
        weights = self.weights_()
        for i in range(len(weights)):
            round_weights.append(round(weights[i],1))
        return round_weights
    
    def minimize_(self):
        cons = ({'type':'eq','fun':self.check_sum})
        bounds = tuple((0, 1) for stocks in range(len(self.select())))
        init_guess = self.round_w()
        opt_results = minimize(self.neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        return opt_results
    
    def final(self):
        opt_results = self.minimize_()
        my_weights = np.array(opt_results.x)
        weights_alloc = dict(zip(self.select(),my_weights))
        return weights_alloc






    # def weights_(self):
    #     weights = np.array(np.random.random(len(self.get_selected_assets())))

    #     print('normalised weights :')
    #     weights = weights/np.sum(weights)
    #     return weights
        
    # def get_ret_vol_sr(self,weights,log_return): 
    #     weights = np.array(weights)
    #     ret = np.sum(log_return.mean() * weights) * 252
    #     vol = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252,weights)))
    #     sr = ret/vol 
    #     return np.array(sr)

    # def neg_sharpe(self,weights): 
    #     return self.get_ret_vol_sr(weights,self.log_returns)*-1

    # def check_sum(self,weights): 
    #     return np.sum(weights) - 1
    
    # def round_w(self):
    #     round_weights = []
    #     weights = self.weights_()
    #     for i in range(len(weights)):
    #         round_weights.append(round(weights[i],1))
    #     return round_weights
    
    # def minimize_(self):
    #     cons = ({'type':'eq','fun':self.check_sum})
    #     bounds = tuple((0, 1) for stocks in range(len(self.get_selected_assets())))
    #     init_guess = self.round_w()
    #     opt_results = minimize(self.neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    #     return opt_results
    
    # def final(self):
    #     opt_results = self.minimize_()
    #     my_weights = np.array(opt_results.x)
    #     weights_alloc = dict(zip(self.get_selected_assets(),my_weights))
    #     return weights_alloc

    # def show_mean_variance(self):
    #     weights=self.weights()
    #     returns=self.returns
    #     portfolio_return = np.sum(returns.mean() * weights) * self.NUM_TRADING_DAYS
    #     portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * self.NUM_TRADING_DAYS, weights)))
    #     print('Expected portfolio mean: ', portfolio_return)
    #     print('Expected portfolio volatility: ', portfolio_volatility)
    #     print("sharpe ratio of a portfolio is ", portfolio_return/portfolio_volatility)
        
        
        
        





