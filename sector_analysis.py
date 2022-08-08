from itertools import groupby
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

    sector_variable = data.columns[data.columns.str.contains(pat = 'industry')].values[0]
   
    st.title("Looking at a Specific Industry Sector")
    industry_selector = st.selectbox("Select the Industry you want to look at", data[sector_variable].unique())
    window_number = st.number_input("Select how many cells you want in the window", 10)
    data['rolling_std'] = data[data[sector_variable]==industry_selector]['weighted_return'].rolling(window_number).std()
    data['rolling_skew'] = data[data[sector_variable]==industry_selector]['weighted_return'].rolling(window_number).skew()
    data['rolling_kurt'] =  data[data[sector_variable]==industry_selector]['weighted_return'].rolling(window_number).kurt()
    fig0 = px.line(data[data[sector_variable]==industry_selector], x = 'date', y = 'weighted_return')
    fig1 = px.line(data, x = 'date', y = 'rolling_std')
    fig2 = px.line(data, x = 'date', y = 'rolling_skew')
    fig3 = px.line(data, x = 'date', y = 'rolling_kurt')
    st.subheader("Weighted Return Over Time")
    st.write(fig0)
    st.subheader("Standard Deviation Over Time")
    st.write(fig1)
    st.subheader("Skewness Over Time")
    st.write(fig2)
    st.subheader("Kurtosis Over Time")
    st.write(fig3)


    # industry_selector = st.selectbox("Select the Industry you want to look at", data.industry_sector.unique())
    # filtered_data = data.loc[(data['industry_sector']==industry_selector)]
    # window_number = st.number_input("Select how many cells you want in the window", 10)
    # # start_date = st.date_input('Start date', data['rfqdate'].min())
    # # end_date = start_date + window_nu
    # # rolling_window = 

    # filtered_data['rolling_std'] = filtered_data.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(window_number).std())
    # px.line(x = filtered_data['date'], y = filtered_data['rolling_std'] )
    # data['rolling_skew'] = filtered_data.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(window_number).skew())
    # data['rolling_mean'] = filtered_data.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(window_number).mean())
    # data['rolling_skew'] = filtered_data.groupby('industry_sector')['weighted_return'].apply(pd.DataFrame.kurt).rolling
    # cutoff1 = norm.ppf(0.05, data['rolling_mean'], data['rolling_std'])
    # data['rolling_VaR'] = data['weighted_return'] - cutoff1

  #  filtered_data.groupby('industry_sector')['weighted_return'].apply(lambda x : x.rolling(window_number).kurtosis())
    #st.write(data['rolling_kurt'].head())
    #st.write(data['rolling_std'])
    # pd.rolling_apply(data, window_number, lambda x: np.prod(1+x) -1)
# df2 = data.loc[(data.event == event_name)].groupby('id').agg({'timestamp': 'min'})

    # data.groupby['industry_sector']


if __name__ == '__main__':
	main()