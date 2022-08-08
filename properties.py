# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:24:35 2022

@author: NORRIKA
"""
import signal
import pandas as pd
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    with st.sidebar:
        #uploading file
        uploaded_file = st.file_uploader("Select the file you would like to interpret", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv('BondSignalData.csv')
            st.info("Default is BondSignalData.csv")

        #filter data by time
        data['rfqdatetime'] = pd.to_datetime(data.request_time)
        data['rfqdate']=data['rfqdatetime'].dt.date
        start_date = st.date_input('Start date', data['rfqdate'].min())
        end_date = st.date_input('End date',  data['rfqdate'].max())
        filtered_data = data.loc[(data['rfqdate'] >= start_date) & (data['rfqdate'] <= end_date)]
    
        if start_date < end_date:
            if start_date >= data['rfqdate'].min():
                if end_date <= data['rfqdate'].max():
                    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
                else:
                    st.error('Error: No data beyond:')
                    st.write(data['rfqdate'].max())
            else:
                st.error('Error: No data before:')
                st.write(data['rfqdate'].min())
        else:
            st.error('Error: End date must fall after start date.')
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

    st.title("Looking at the Different Signals")
    property_selector = st.selectbox("Select Property", ("Liquidity Score", "Quantity", "Maturity", "Industry Sector"))

    if property_selector == "Liquidity Score":
        st.subheader("Liquidity Score")
        #distplot
        fig1 = px.histogram(filtered_data, x='liq_score')
        st.write(fig1)
        filtered_data['binnedliq_score']= pd.qcut(filtered_data['liq_score'], q=8)
        filtered_data['binnedliq_score'].cat.categories = [f'{i.left} to {i.right}' for i in filtered_data['binnedliq_score'].cat.categories]
        liquidity_selector = st.selectbox("Select the range the RFQ liquidity score lies between", (filtered_data['binnedliq_score'].cat.categories))
        signal_selector = st.selectbox("Select Signal to look at OLS data", ("signal3", "signal2", "signal1"))
        x = filtered_data[filtered_data['binnedliq_score']==liquidity_selector][signal_selector]
        y = filtered_data[filtered_data['binnedliq_score']==liquidity_selector]['move_1D']

        x = sm.add_constant(x)
    
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x) 

    
        variable_coefs = model.params
        confidence_interval= model.conf_int(alpha=0.05)
        p_values= model.pvalues
        r_squared= model.rsquared
    
        sub_df = pd.DataFrame()
        sub_df['features'] = ['const', signal_selector]
        sub_df.set_index('features', inplace=True)
        sub_df['coef'] = list(variable_coefs)
        sub_df['conf_a'] = list(confidence_interval[0])
        sub_df['conf_b'] = list(confidence_interval[1])
        sub_df['p_values'] = list(p_values)
        sub_df['r_squared'] = r_squared
    
        st.write("Liquidity Score in", liquidity_selector, signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['binnedliq_score'] == liquidity_selector]['move_1D'].mean())

    elif property_selector == "Quantity":
        st.subheader("Quantity of Bonds")
        #distplot
        fig1 = px.histogram(filtered_data, x='quantity_bonds')
        st.write(fig1)
        filtered_data['binnedquantity_bonds']= pd.qcut(filtered_data['quantity_bonds'], q=5)
        filtered_data['binnedquantity_bonds'].cat.categories = [f'{i.left} to {i.right}' for i in filtered_data['binnedquantity_bonds'].cat.categories]
        quantity_selector = st.selectbox("Select the range the quantity of the RFQ lies in", (filtered_data['binnedquantity_bonds'].cat.categories))
        signal_selector = st.selectbox("Select Signal to look at OLS data", ("signal3", "signal2", "signal1"))
        x = filtered_data[filtered_data['binnedquantity_bonds']==quantity_selector][signal_selector]
        y = filtered_data[filtered_data['binnedquantity_bonds']==quantity_selector]['move_1D']

        x = sm.add_constant(x)
    
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x) 

    
        variable_coefs = model.params
        confidence_interval= model.conf_int(alpha=0.05)
        p_values= model.pvalues
        r_squared= model.rsquared
    
        sub_df = pd.DataFrame()
        sub_df['features'] = ['const', signal_selector]
        sub_df.set_index('features', inplace=True)
        sub_df['coef'] = list(variable_coefs)
        sub_df['conf_a'] = list(confidence_interval[0])
        sub_df['conf_b'] = list(confidence_interval[1])
        sub_df['p_values'] = list(p_values)
        sub_df['r_squared'] = r_squared
    
        st.write("Quantity =", quantity_selector, signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['binnedquantity_bonds'] == quantity_selector]['move_1D'].mean())

    elif property_selector ==  "Maturity":
        st.subheader("Maturity Bucket")

        filtered_data['maturity'] = filtered_data.mat_bucket.str.strip("B")
        filtered_data['maturity']=filtered_data.maturity.astype(int)
         #distplot
        fig1 = px.histogram(filtered_data, x='maturity')
        st.write(fig1)
        #data['binned_maturity']= pd.qcut(data['maturity'], q=5)
        #data['binned_maturity'].cat.categories = [f'{i.left} to {i.right}' for i in data['binned_maturity'].cat.categories]
        maturity_selector = st.selectbox("Select the maturity of the RFQ", (filtered_data['maturity'].unique()))
        signal_selector = st.selectbox("Select Signal to look at OLS data", ("signal3", "signal2", "signal1"))
        x = filtered_data[filtered_data['maturity']==maturity_selector][signal_selector]
        y = filtered_data[filtered_data['maturity']==maturity_selector]['move_1D']

        x = sm.add_constant(x)
    
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x) 

    
        variable_coefs = model.params
        confidence_interval= model.conf_int(alpha=0.05)
        p_values= model.pvalues
        r_squared= model.rsquared
    
        sub_df = pd.DataFrame()
        sub_df['features'] = ['const', signal_selector]
        sub_df.set_index('features', inplace=True)
        sub_df['coef'] = list(variable_coefs)
        sub_df['conf_a'] = list(confidence_interval[0])
        sub_df['conf_b'] = list(confidence_interval[1])
        sub_df['p_values'] = list(p_values)
        sub_df['r_squared'] = r_squared
    
        st.write("Maturity =", maturity_selector, "years", signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['maturity'] == maturity_selector]['move_1D'].mean())

    elif property_selector == "Industry Sector":
        st.subheader("Industry Sector")
        #distplot
        fig1 = px.histogram(filtered_data, x='industrySector')
        st.write(fig1)
        filtered_data['sector_category']= filtered_data.industrySector.astype('category')
        #data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=8)
        #data['binnedquantity_bonds'].cat.categories = [f'{i.left} to {i.right}' for i in data['binnedquantity_bonds'].cat.categories]
        industry_selector = st.selectbox("Select the industry sector of the RFQ", (filtered_data['sector_category'].unique()))
        signal_selector = st.selectbox("Select Signal to look at OLS data", ("signal3", "signal2", "signal1"))
        x = filtered_data[filtered_data['industrySector']==industry_selector][signal_selector]
        y = filtered_data[filtered_data['industrySector']==industry_selector]['move_1D']

        x = sm.add_constant(x)
    
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x) 

    
        variable_coefs = model.params
        confidence_interval= model.conf_int(alpha=0.05)
        p_values= model.pvalues
        r_squared= model.rsquared
    
        sub_df = pd.DataFrame()
        sub_df['features'] = ['const', signal_selector]
        sub_df.set_index('features', inplace=True)
        sub_df['coef'] = list(variable_coefs)
        sub_df['conf_a'] = list(confidence_interval[0])
        sub_df['conf_b'] = list(confidence_interval[1])
        sub_df['p_values'] = list(p_values)
        sub_df['r_squared'] = r_squared
    
        st.write(industry_selector, signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['industrySector'] == industry_selector]['move_1D'].mean())
    pass