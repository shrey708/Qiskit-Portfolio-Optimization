import yfinance as yf
import numpy as np

class Data_load:
    
    def __init__(self, assets, start_date, end_date, ) -> None:
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        
        
    def get_data(self):
        stocks_data = yf.download(self.assets, start = self.start_date, end = self.end_date)['Adj Close']

        log_return = np.log(stocks_data/stocks_data.shift(1))
        
        # return stocks_data, num_assets, mu, sigma
        return log_return
    
    
