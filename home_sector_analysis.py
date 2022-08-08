import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
import plotly.express as px 

def main():
    with st.sidebar:
        #uploading file
        uploaded_file = st.file_uploader("Select the file you would like to interpret", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv('port_returns_jason.csv')
            st.info("Default is port_returns_jason.csv")
    st.title("Looking at All Industry Sectors at Once")
    sector_variable = data.columns[data.columns.str.contains(pat = 'industry')].values
    window_number = st.number_input("Select how many cells you want in the window", 10)
    data['rolling_std'] = data.groupby('industry_sector_x')['weighted_return'].apply(lambda x : x.rolling(window_number).std())
    data['rolling_skew'] = data.groupby('industry_sector_x')['weighted_return'].apply(lambda x : x.rolling(window_number).skew())
    data['rolling_kurt'] =  data.groupby('industry_sector_x')['weighted_return'].apply(lambda x : x.rolling(window_number).kurt())
   # data['rolling_avg'] = data.groupby('industry_sector_x')['weighted_return'].apply(lambda x : x.rolling(window_number))
    fig0 = px.line(data, x = 'date', y = 'weighted_return', color='industry_sector_x')
    fig1 = px.line(data, x = 'date', y = 'rolling_std', color='industry_sector_x')
    fig2 = px.line(data, x = 'date', y = 'rolling_skew', color='industry_sector_x')
    fig3 = px.line(data, x = 'date', y = 'rolling_kurt', color='industry_sector_x')
    st.subheader("Weighted Return Over Time")
    st.write(fig0)
    st.subheader("Standard Deviation Over Time")
    st.write(fig1)
    st.subheader("Skewness Over Time")
    st.write(fig2)
    st.subheader("Kurtosis Over Time")
    st.write(fig3)

    # portfolio_returns['rolling_stddev'] = portfolio_returns.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(n_periods).std())

    # portfolio_returns['rolling_skew'] = portfolio_returns.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(n_periods).skew())

    # portfolio_returns['rolling_kurt'] = portfolio_returns.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(n_periods).kurt())

    # portfolio_returns['rolling_drawdown'] = portfolio_returns.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(n_periods).max() - x.rolling(n_periods).min())