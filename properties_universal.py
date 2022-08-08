import signal
from unicodedata import category
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

        signal_variables = st.multiselect("Select the Signal Variables", data.columns)
        response_variables = st.multiselect("Select the Response Variables", data.columns)
        

    st.title("Looking at the Different Signals")
    property_selector = st.selectbox("Select Property", data.columns, index =3)
    move_selector =  st.selectbox("Select Movement", response_variables, index =0)
    st.subheader(property_selector)
    fig1 = px.histogram(filtered_data, x=property_selector)
    st.write(fig1)
    #filtered_data['binned']= pd.qcut(filtered_data[property_selector], q=bin_size)

    if filtered_data[property_selector].dtype != object:
        bin_size =  st.number_input("Select the number of bins", 5)
        filtered_data['binned']= pd.qcut(filtered_data[property_selector].astype('category'), q=bin_size)
        filtered_data['binned'].cat.categories = [f'{i.left} to {i.right}' for i in filtered_data['binned'].cat.categories]

        bin_selector = st.selectbox("Select the bin the RFQ lies in", (filtered_data['binned'].cat.categories))
        signal_selector = st.selectbox("Select Signal to look at OLS data", signal_variables, index =0)
        x = filtered_data[filtered_data['binned']==bin_selector][signal_selector]
        y = filtered_data[filtered_data['binned']==bin_selector][move_selector]

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
        
        st.write(bin_selector, signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['binned'] == bin_selector][move_selector].mean())   
    else: 
        filtered_data['binned'] = filtered_data[property_selector].astype('category')
        bin_selector = st.selectbox("Select the bin the RFQ lies in", (filtered_data['binned'].unique()))
        signal_selector = st.selectbox("Select Signal to look at OLS data", signal_variables, index =0)
        x = filtered_data[filtered_data['binned']==bin_selector][signal_selector]
        y = filtered_data[filtered_data['binned']==bin_selector][move_selector]

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
        
        st.write(bin_selector, signal_selector)
        st.write(sub_df)
        st.write("The average movement after 1 day is", filtered_data[filtered_data['binned'] == bin_selector][move_selector].mean())   






  #  below doesn't work 
    # filtered_data['binned', property_selector]= pd.qcut(filtered_data[property_selector], q=5)
    # filtered_data['binned', property_selector].cat.categories = [f'{i.left} to {i.right}' for i in filtered_data['binned', property_selector].cat.categories]
    # filtered_data['binned', property_selector].head()
    # liquidity_selector = st.selectbox("Select the range the RFQ liquidity score lies between", (filtered_data['binnedliq_score'].cat.categories))
    # signal_selector = st.selectbox("Select Signal to look at OLS data", ("signal3", "signal2", "signal1"))
    # x = filtered_data[filtered_data['binnedliq_score']==liquidity_selector][signal_selector]
    # y = filtered_data[filtered_data['binnedliq_score']==liquidity_selector]['move_1D']