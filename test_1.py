import data_load
import weights_alloc
import numpy as np
import streamlit as st 


# assets = ["AMZN", "BAC", "CVX", "SPY"]
# start_date  = '2013-01-01'
# end_date = '2021-12-31'

# o = data_load.Data_load(assets,start_date,end_date)
# stocks_data = o.get_data_from_yf()
# mu,sigma,num_assets = o.variables(stocks_data)

# selected_assets = [1, 0, 0, 1]

# q = weights_alloc.Weights_allocation(stocks_data,selected_assets,mu)

# weights = q.weights_()
# print(weights)
# print()
# log_return = q.log()
# print(q.log())
# print()
# print(q.get_ret_vol_sr(weights,log_return))
# print()
# print(q.neg_sharpe(weights))
# print()
# print(q.check_sum(weights))
# print()
# print(q.round_w())
# print()
# print(q.minimize_())
# print()
# print(q.final())

import streamlit as st
import sqlite3 
from st_btn_select import st_btn_select
import hashlib

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
import hydralit as hy

app = hy.HydraApp(title='Portfolio Optimization App')

@app.addapp()
def my_home():
    hy.info('Login Page')

 # Create an empty container
    placeholder = st.empty()

    # actual_email = "email"
    # actual_password = "password"
    # username = st.text_input("User Name")
    # password = st.text_input("Password",type='password')
        

    #st.title("Welcome")
    st.title("Portfolio Optimization login page")
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
        st.title("PORTFOLIO OPTIMIZATION")
        

    elif submit and username != username and password != password:
        st.error("Login failed")
    else:
        pass

@app.addapp()
def user_manual():
 hy.info('User Manual') 


#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()








