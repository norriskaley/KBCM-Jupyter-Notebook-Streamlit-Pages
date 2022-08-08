from tkinter.messagebox import YES
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px 
import plotly.graph_objects as go
from statsmodels.tools.eval_measures import rmse

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
            else:
                st.error('Error: No data before:')
                st.write(data['rfqdate'].min())
        else:
            st.error('Error: End date must fall after start date.')
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        signal_variables = st.multiselect("Select the Signal Variables", data.columns)
        response_variables = st.multiselect("Select the Response Variables", data.columns)

    st.title("Creating an Indicator Variable")
    st.write("If you believe an indicator variable(s) would make the Linear Regression model more accurate, this page will allow you to add those for the signal of your choosing.")
    signal_selector = st.selectbox("Select Signal",  signal_variables, index =0)
    move_selector =  st.selectbox("Select Movement", response_variables, index =0)
    # indicator_type = st.radio("Would you like the indicator to be at specific value or over a range of values?", ("Specific", "Range"))
    # if indicator_type == "Range":
        # left_endpoint =  st.text_input("Enter the left endpoint value", -1)
        # left_included = st.radio("Would you like this endpoint to be included in the range?", ("Yes", "No"))
        # if left_included == "No":
        #     gl
        # else:

        # right_endpoint = st.text_input("Enter the right endpoint value", -1)
        # right_included = st.radio("Would you like this endpoint to be included in the range?", ("Yes", "No"))
        # if right_included == "No":
    # else:
    
    signal_add_indicator = st.text_input("Enter the value at which you want the indicator variable", 1)
    filtered_data['signal_equal_indicator'] = filtered_data[signal_selector].apply(lambda x: 1 if x == float(signal_add_indicator) else 0)
    st.success("Indicator Variable Created")
    another_indicator = st.radio("Would you like to create another indicator variable?", ("No", "Yes"))
    if another_indicator == "Yes":
        signal_add_indicator2 = st.text_input("Enter the value at which you want another indicator variable", 1)
        filtered_data['signal_equal_indicator2'] = filtered_data[signal_selector].apply(lambda x: 1 if x == float(signal_add_indicator2) else 0)
        col1, col2 = st.columns(2)
        with col1:
            x = filtered_data[[signal_selector, 'signal_equal_indicator', 'signal_equal_indicator2']]
            y = filtered_data[move_selector]
    
            x = sm.add_constant(x)

            model = sm.OLS(y, x).fit()
            predictions = model.predict(x) 
            filtered_data['model_params'] = model.params[0]

            print_model = model.summary()
            
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write("R-squared tell us how much percentage variation in the dependent variable can be explained by the independent variables. In general, the larger the R-squared value of a regression model, the better the explanatory variables(x) are able to predict the value of the response variable(y). In this case,", model.rsquared * 100, "% of the variation can be explained.")
            if model.rsquared < 0.4:
                st.error("R-squared is weak")
            elif model.rsquared < 0.7:
                st.warning("R-squared is moderate")
            else:
                st.success("R-squared is strong")
            st.write("P>|t| is the p-value associated with the model coefficients. A p-value is statistically sigifciant if it's less than 0.05. In this case, we can reject the null hypothesis as a relationship between the independent and dependent variables exists.")
            if model.pvalues[0] < 0.05:
                st.success("The constant p-value is statistically significant")
            else:
                st.error("The constant p-value is greater than 0.05, therefore we cannot reject the null hypothesis.")
            if model.pvalues[1] < 0.05:
                st.success("The signal p-value is statistically significant")
            else:
                st.error("The signal p-value is greater than 0.05, therefore we cannot reject the null hypothesis.")
            if model.pvalues[2] < 0.05:
                st.success("The first indicator p-value is statistically significant")
            else:
                st.error("The first indicator p-value is greater than 0.05, therefore we cannot reject the null hypothesis.")
            if model.pvalues[3] < 0.05:
                st.success("The second indicator p-value is statistically significant")
            else:
                st.error("The second indicator p-value is greater than 0.05, therefore we cannot reject the null hypothesis.")
        with col2:
            st.write(print_model)
    else:
        col1, col2 = st.columns(2)
        with col1:
            x = filtered_data[[signal_selector, 'signal_equal_indicator']]
            y = filtered_data[move_selector]

            x = sm.add_constant(x)

            model = sm.OLS(y, x).fit()
            predictions = model.predict(x) 
            filtered_data['model_params'] = model.params[0]

            print_model = model.summary()
            
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write("R-squared tell us how much percentage variation in the dependent variable can be explained by the independent variables. In general, the larger the R-squared value of a regression model, the better the explanatory variables(x) are able to predict the value of the response variable(y). In this case,", model.rsquared * 100, "% of the variation can be explained.")
            if model.rsquared < 0.4:
                st.error("R-squared is weak")
            elif model.rsquared < 0.7:
                st.warning("R-squared is moderate")
            else:
                st.success("R-squared is strong")
            st.write("P>|t| is the p-value associated with the model coefficients. A p-value is statistically sigifciant if it's less than 0.05. In this case, we can reject the null hypothesis as a relationship between the independent and dependent variables exists.")
            if model.pvalues[0] < 0.05:
                st.success("The constant p-value is statistically significant")
            else:
                st.error("The constant p-value is greater than 0.05 therefore we cannot reject the null hypothesis.")
            if model.pvalues[1] < 0.05:
                st.success("The signal p-value is statistically significant")
            else:
                st.error("The signal p-value is greater than 0.05 therefore we cannot reject the null hypothesis.")
            if model.pvalues[2] < 0.05:
                st.success("The first indicator p-value is statistically significant")
            else:
                st.error("The first indicator p-value is greater than 0.05 therefore we cannot reject the null hypothesis.")
        with col2:
            st.write(print_model)
        

