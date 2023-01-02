
from data_load import Data_load
from portfolio_bitstring import portfolio_opt
from weights_allocation import opt_weights, get_portfolio_sharpe
import datetime as datetime 
# from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import hydralit as hy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sqlite3
import yfinance as yf
import altair as alt
# import plotly.express as px

app = hy.HydraApp(title= ' Portfolio Optimization App ')





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
        
        if self.past_years == '5 Years':
            past_years = 5
        
        elif self.past_years == '10 Years':
            past_years = 10
            
        elif self.past_years == '15 Years':
            past_years = 15
            
        elif self.past_years == '20 Years':
            past_years = 20
            
        start_date  = datetime.date.today() - relativedelta(years=past_years)
        print("start date is ", start_date)
        data = Data_load(self.assets, start_date, end_date)
        log_returns= data.get_data()
        
        index_data = yf.download("^DJI", start_date, end_date)["Adj Close"]
        index_data = np.log(index_data/index_data.shift(1))

        # print("returns are ", log_returns)
        opt_bit = portfolio_opt(self.assets, log_returns, self.budget, self.device, trading_days = 252)
        
        
        if self.algorithm ==  "QAOA with cobyla":
            opt_bitstring  = opt_bit.get_solution_using_qaoa_cobyla()
        elif self.algorithm ==  "QAOA with SPSA":
            opt_bitstring  = opt_bit.get_solution_using_qaoa_spsa()
        elif self.algorithm ==  "VQE with Cobyla":
            opt_bitstring  = opt_bit.get_solution_using_vqe_cobyla()
        elif self.algorithm ==  "VQE with SPSA":
            opt_bitstring  = opt_bit.get_solution_using_vqe_spsa()
        
        
        
        # opt_bitstring 
        
        my_weight  = opt_weights(log_returns, opt_bitstring)
        my_weights = my_weight.optimize_weights()
        # my_weights
        my_stocks = list(my_weights.keys())
        
        portfolio_data = yf.download(my_stocks, start_date, end_date)["Adj Close"]
        portfolio_data = np.log(portfolio_data/portfolio_data.shift(1))
        portfolio_data["portfolio"] = portfolio_data.mean(axis =1)
        
        portfolio_data = portfolio_data["portfolio"]
        # portfolio_data = pd.DataFrame(portfolio_data, columns= ["Date", "Values"])
        
        print(portfolio_data)
        sharpe = get_portfolio_sharpe(log_returns, my_weights)
        sharpe_ratio = sharpe.get_sharpe_ratio()
        
        return my_weights, sharpe_ratio, index_data, portfolio_data
        



def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management

conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data



#when we import hydralit, we automatically get all of Streamlit


@app.addapp()
def login():
    hy.info('Login Page')

 # Create an empty container
    placeholder = st.empty()

    # actual_email = "email"
    # actual_password = "password"
    # username = st.text_input("User Name")
    # password = st.text_input("Password",type='password')
        

    #st.title("Welcome")
    st.title(" Quantum Powered Portfolio Optimization ")
    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        username = st.text_input("User Name")
        password = st.text_input("Password",type='password')
        # if st.button("Login"):
            # if password == '12345':
        create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        submit = st.form_submit_button("Login")

    if submit and username == username and password == password:
        # If the form is submitted and the email and password are correct,
        # clear the form/container and display a success message
        placeholder.empty()
        
        st.success("Login successful")
        st.title("WELCOME")
        st.title(" Portfolio Optimization Dashbboard")
        # protein_core_app()
        

    elif submit and username != username and password != password:
        st.error("Login failed")
    else:
        pass




@app.addapp()
def user_manual():
    hy.info(' User Manual ') 
    
    st.subheader(" Get the quantum power to get optimal portfolio for your financial investments")
    
    st.write("\n \n \n")
    
    st.write(" sometime it might take few more seconds to load the data, and to show you the optimal result, kindly wait for a minute. ")
    st.subheader(" steps to use our system ")
    
    
    
    st.write(" we load our data on per minute basis, so after every minute you can generate profit on every trade executed")
    
    st.write("\n \n \n")
    st.write("1. Login to our system")
    st.write("2. select the stocks that you want ")
    st.write("3. select the values for other inputs asked ")
    st.write("4. click on the button to get you the results ")

    st.write("5. read the output and execute the trade accoridngly ")
    st.write("6. earn profit and make life happier  ")






@app.addapp()
def portfolio_core_app():
    
    st.title ("  Portfolio Optimization  ")

    st.subheader(" Portfolio Optimization app  ")

    investment_amount =  st.sidebar.number_input('investment ', min_value=10000, step=500)

    algorithm = st.sidebar.radio(
            "Algorithm ",
            key="visibility",
            options=[ "QAOA with cobyla", "QAOA with SPSA", "VQE with Cobyla", "VQE with SPSA"],
    )
    # ['AED', 'ARS', 'AUD', 'BGN', 'BRL', 'BSD', 'CAD', 'CHF', 'CLP', 'CNY', 'COP', 'CZK', 'DKK',  'DOP', 'EGP', 'GBP', 'HKD', 'HRK', 'HUF', 'IDR', 'ILS', 'INR', 'ISK', 'JPY', 'MVR', 'MXN', 'MYR', 'NOK', 'NZD', 'RUB', 'SAR', 'SEK', 'SGD', 'THB', 'TRY', 'TWD', 'UAH', 'USD', 'UYU', 'ZAR']
    stocks = pd.DataFrame({'labels':['WMT','WBA', 'VZ', 'V', 'UNH', 'TRV', 'PG', 'NKE', 'MSFT',  'MRK',  'MMM', 'MCD', 'KO', 'JPM', 'JNJ', 'INTC', 'IBM', 'HON', 'HD', 'GS', 'DOW', 'DIS', 'CVX', 'CSCO', 'CRM', 'CAT', 'BA', 'AXP', 'AMGN', 'AAPL']})


    index = st.sidebar.radio(
                        " Stocks Market Index ", 
                        options = [' Dow 30 ']
    )

    assets = st.sidebar.multiselect('Select the Currencies you want to trade with', 
                                    options=list(stocks['labels']),
                                    default =["JPM", "GS", "MSFT", "AAPL", "WMT", "MCD"])

        
    past_data = st.sidebar.radio(
                        " Past years ", 
                        options = ['5 Years', '10 Years', '15 Years', '20 Years']
    )

    device = st.sidebar.radio(
                        " Simulator device ", 
                        options = [' QASM Simulator ']
    )

    trading_days = st.sidebar.radio(" Trading days  ", 
                        options = [' 252 ']
    )


    budget =  st.sidebar.number_input('no of stocks to be selected in portfolio', min_value=len(assets)//4, max_value=len(assets), value=len(assets)//2, step=1)




    if st.sidebar.button('Get optimal portfolio '):
        
        
        
        assets = sorted(assets)
        print(assets)
        ## definfing a button, which after pressing will create a qubo and will give optiml path to be used by an user.
        portfolio_app = run_app(index,  assets, budget, past_data, algorithm, device, trading_days)
        portfolio_output = portfolio_app.app()
        my_weights, sharpe_ratio, index_data, portfolio_data = portfolio_output
        print("data type is ", type(portfolio_data))
        print("data typpppeess is ", type(index_data))
        # chart_df = portfolio_data.merge(index_data,left_index=True, right_index=True)
        print(sharpe_ratio)
        portfolio_val = portfolio_data.mean()
        index_val = index_data.mean()
        
        portfolio_index_ratio = portfolio_val / index_val
        print("ratio ", portfolio_index_ratio)
        
        my_stocks = list(my_weights.keys())
        weights = list(my_weights.values())
        
        weights = [item * 100 for item in weights]
        
        print(" \n \n ")
        print(my_stocks)
        print(weights)
        
        holdngs_data = {"Assets":my_stocks, "Weights(%)": weights } #,"Money($)":allocated_money}
        holdngs_df = pd.DataFrame(holdngs_data)
        # print(summary_df)
        sharpe_ratio_df = pd.DataFrame(sharpe_ratio.items(), columns=['index', 'value'])
        
        
        col_1, col_2= st.columns([1.3,2])
        with col_1:
    
            st.subheader(" Weights Allocation ")
            
            st.table(holdngs_df.sort_values(by=["Weights(%)"],ascending=False))
            
            fig, ax = plt.subplots()
            ax.pie(weights,labels=my_stocks)
            st.pyplot(fig)
            
            # st.subheader("index graph")
            # st.line_chart(index_data)
            # st.table(chart_df)
            
            st.write(" \n  \n ")
    
            
            st.subheader("Portfolio quality ")
            # st.table(sharpe_ratio_df)
        
        st.subheader(" Portfolio vs Index graph ")
        chart_df = pd.concat([portfolio_data, index_data], axis=1)
        print("tyyytpppesss 0000 ", type(chart_df))
        st.line_chart(chart_df)
        
        st.write("our portfolio beats index by " +str(portfolio_index_ratio)+ " times.. ")
        st.write(" So, you can see our portfolio is beating the Dow30 index.. ")


            
                
        # allocated_money = np.array(list(investment_amount*(weights/100)))
        
        # allocated_money = np.round(investment_amount*weights/100,2)
        # allocated_money = [item * 100 for item in allocated_money]
        # print("allocated money ",type(allocated_money))
        
        
        st.write(" \n \n ")
        
        # st.table(portfolio_quality_table)
        



@app.addapp()
def signup():
    hy.info("Create New Account ")
    
    # st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')

    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user,make_hashes(new_password))
        st.success("You have successfully created a Account")
        st.info("Go to Login Menu to login")

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()

# 


