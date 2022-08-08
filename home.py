# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:55:25 2022

@author: NORRIKA
"""
from ast import And
import pandas as pd
import dashboard
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go
import datetime

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

    st.title("RFQ Signal Analysis")
    st.markdown("The dashboard will help a trader to get to know more about the given datasets and its output")

    #pairplot
    st.subheader("Scatterplots to Visualize Correlations")
    st.write("A relationship exists if the points are highly concentrated, moving in the same direction.")
    fig1 = px.scatter_matrix(filtered_data, dimensions=['move_1D','move_3D','signal1', 'signal2', 'signal3'])
    st.write(fig1)

    #heatmap
    st.subheader("Heatmap to Showcase Correlations")
    st.write("The darker the red tone, the more correlated that signal is with the movement data.")
    matrix = filtered_data[['move_1D','move_3D','signal1', 'signal2', 'signal3']]
    def matrix_to_plotly(matrix):
        return {'z':matrix.values.tolist(),
                'x': matrix.columns.tolist(),
                'y': matrix.index.tolist() }
    matrixNew = matrix.corr()
    fig2 = go.Figure(data=go.Heatmap(matrix_to_plotly(matrixNew), colorscale="reds"))
    st.write(fig2)

    if filtered_data['move_1D'].corr(filtered_data['signal3']) > filtered_data['move_1D'].corr(filtered_data['signal2']) and filtered_data['move_1D'].corr(filtered_data['signal3']) > filtered_data['move_1D'].corr(filtered_data['signal1']):
        st.success("Signal 3 is the best signal")
    elif filtered_data['move_1D'].corr(filtered_data['signal2']) > filtered_data['move_1D'].corr(filtered_data['signal3']) and filtered_data['move_1D'].corr(filtered_data['signal2']) > filtered_data['move_1D'].corr(filtered_data['signal1']):
        st.success("Signal 2 is the best signal")
    elif filtered_data['move_1D'].corr(filtered_data['signal1']) > filtered_data['move_1D'].corr(filtered_data['signal3']) and filtered_data['move_1D'].corr(filtered_data['signal1']) > filtered_data['move_1D'].corr(filtered_data['signal2']):
        st.success("Signal 1 is the best signal")
    else:
        pass
    # st.write("The average movement after 1 day is", filtered_data['move_1D'].mean())
    # st.write("The average movement after 3 days is", filtered_data['move_3D'].mean())
    
if __name__ == '__main__':
	main()
