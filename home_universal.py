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
        
        
        request_time = st.selectbox("Select the column with Request Times", data.columns)
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
                    st.write("The data is being displayed up until", str(data['rfqdate'].max()), '.')
            else:
                st.error('Error: No data before:')
                st.write(data['rfqdate'].min())
        else:
            st.error('Error: End date must fall after start date.')
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        signal_variables = st.multiselect("Select the Signal Variables", data.columns)
        response_variables = st.multiselect("Select the Response Variables", data.columns)

    st.title("RFQ Signal Analysis")
    st.markdown("The dashboard will help a trader to get to know more about the given datasets and its output")
    x_and_y_variables = signal_variables + response_variables
    #st.multiselect("Select the Response and Predicting Variables", signal_variables and response_variables)
    if len(x_and_y_variables) == 0:
        st.error("Please select variables above")
    else:
        col1, col2 =st.columns(2)
        with col1:
            st.subheader("Scatterplots to Visualize Correlations")
            st.write("A relationship exists if the points are highly concentrated, moving in the same direction.")
            
            fig1 = px.scatter_matrix(filtered_data, dimensions=x_and_y_variables)
            st.write(fig1)

        with col2:
            st.subheader("Heatmap to Showcase Correlations")
            st.write("The darker the red tone, the more correlated the variables are.")
            matrix = filtered_data[x_and_y_variables]
            def matrix_to_plotly(matrix):
                return {'z':matrix.values.tolist(),
                        'x': matrix.columns.tolist(),
                        'y': matrix.index.tolist() }
            matrixNew = matrix.corr()
            fig2 = go.Figure(data=go.Heatmap(matrix_to_plotly(matrixNew), colorscale="reds"))
            st.write(fig2)
        
        def get_redundant_pairs(matrix):
            '''Get diagonal and lower triangular pairs of correlation matrix'''
            pairs_to_drop = set()
            cols = matrix.columns
            for i in range(0, matrix.shape[1]):
                for j in range(0, i+1):
                    pairs_to_drop.add((cols[i], cols[j]))
            return pairs_to_drop

        def get_top_abs_correlations(matrix, n=1):
            au_corr = matrix.corr().abs().unstack()
            labels_to_drop = get_redundant_pairs(matrix)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            # st.write(au_corr[0:n].reset_index())
            return au_corr[0:n].reset_index()
            
        #st.write("The Best Signal in terms of Correlation")
        if len(x_and_y_variables) < 2:
            pass
        # elif x_and_y_variables.dtypes() != 
        else:
            #x_and_y_variables.astype(int)
            st.write("The most highly correlated variables are", get_top_abs_correlations(matrix).loc[0, 'level_0'], 'and', get_top_abs_correlations(matrix).loc[0, 'level_1'], "with a correlation of", get_top_abs_correlations(matrix).loc[0, 0].round(4))
        #st.write(get_top_abs_correlations(matrix)[])


if __name__ == '__main__':
	main()