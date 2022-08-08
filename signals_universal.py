import pandas as pd
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.io as pio
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
        regression_selector = st.sidebar.radio("What type of regression would you like to look at?", ("Linear", "Quadratic", "Cubic"))
        
    st.title("Looking at the Different Signals")
    signal_selector = st.selectbox("Select Signal", signal_variables, index =0)
    move_selector =  st.selectbox("Select Movement", response_variables, index =0)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution Plot")
        #distplot
        fig1 = px.histogram(filtered_data, x=signal_selector)
        st.write(fig1)
    with col2:
        if regression_selector == "Cubic":
            st.subheader("Cubic Regression Model")
            x = filtered_data[signal_selector]
            y = filtered_data[move_selector]

            x = sm.add_constant(x)

            m = filtered_data[move_selector]
            signal = filtered_data[signal_selector]
            model = 'm ~ signal + I(signal**2) + I(signal**3)'
            #model= 'move_1D ~ signal3 + I(signal3**2) + I(signal3**3)'
            signal_model = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_model_params'] = signal_model.params[0]

            predictions = signal_model.predict(x) 
            print_model = signal_model.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif regression_selector == "Quadratic":
            st.subheader("Quadratic Regression Model")
            x = filtered_data[signal_selector]
            y = filtered_data[move_selector]

            x = sm.add_constant(x)

            m = filtered_data[move_selector]
            signal = filtered_data[signal_selector]
            model = 'm ~ signal + I(signal**2)'
            #model= 'move_1D ~ signal3 + I(signal3**2)'
            signal_model = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_model_params'] = signal_model.params[0]

            predictions = signal_model.predict(x) 
            print_model = signal_model.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        else: 
            st.subheader("Linear Regression Model")
            x = filtered_data[signal_selector]
            y = filtered_data[move_selector]

            x = sm.add_constant(x)

            signal_model = sm.OLS(y, x).fit()
            predictions = signal_model.predict(x) 
            filtered_data['signal_model_params'] = signal_model.params[0]

            print_model = signal_model.summary()
            #st.write(print_model)
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        # else:
        #     st.subheader("Linear Regression Model")
        #     move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
        #     #LinReg
        #     if move_selector == "1 Day":
                
        #         x = filtered_data[signal_selector]
        #         y = filtered_data[move_selector]

        #         x = sm.add_constant(x)

        #         signal_model = sm.OLS(y, x).fit()
        #         predictions = signal_model.predict(x) 
        #         filtered_data['signal_model_params'] = signal_model.params[0]

        #         print_model = signal_model.summary()
        #         #st.write(print_model)
        #         #graph LinRrg
        #         fig2 = px.scatter(filtered_data, x=signal_selector, y=move_selector)
        #         fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
        #         fig4 = px.line(filtered_data, x=signal_selector, y='signal_model_params', color_discrete_sequence = ['black'])
        #         fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
        #         st.write(fig5)
        #         st.write(print_model)
                
        #     elif move_selector == "3 days":
        #         x = filtered_data[signal_selector]
        #         y = filtered_data['move_3D']

        #         x = sm.add_constant(x)

        #         signal_model = sm.OLS(y, x).fit()
        #         predictions = signal_model.predict(x) 
        #         filtered_data['signal_model_params'] = signal_model.params[0]

        #         print_model = signal_model.summary()
        #         #st.write(print_model)
        #         #graph LinRrg
        #         fig2 = px.scatter(filtered_data, x=signal_selector, y='move_3D')
        #         fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
        #         fig4 = px.line(filtered_data, x=signal_selector, y='signal_model_params', color_discrete_sequence = ['black'])
        #         fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
        #         st.write(fig5)
        #         st.write(print_model)
        #     else:
        #         pass
    with col1:
        st.write("R-squared tell us how much percentage variation in the dependent variable can be explained by the independent variables. In general, the larger the R-squared value of a regression model, the better the explanatory variables(x) are able to predict the value of the response variable(y). In this case,", signal_model.rsquared * 100, "% of the variation can be explained.")
        if signal_model.rsquared < 0.4:
            st.error("R-squared is weak")
        elif signal_model.rsquared < 0.7:
            st.warning("R-squared is moderate")
        else:
            st.success("R-squared is strong")
        st.write("P>|t| is the p-value associated with the model coefficients. A p-value is statistically sigifciant if it's less than 0.05. In this case, we can reject the null hypothesis as a relationship between the independent and dependent variables exists.")
        if signal_model.pvalues[0] < 0.05:
            st.success("The constant p-value is statistically significant")
        else:
            st.error("The constant p-value is greater than 0.05, so we cannot say that the coefficient is nonzero.")
        if signal_model.pvalues[1] < 0.05:
            st.success("The signal p-value is statistically significant")
        else:
            st.error("The signal p-value is greater than 0.05, so we cannot say that the coefficient is nonzero.")
        if len(signal_model.pvalues) < 3:
            pass
        elif signal_model.pvalues[2] < 0.05:
            st.success("The quadratic term p-value is statistically significant")
        elif signal_model.pvalues[2] >= 0.05:
            st.error("The quadratic term p-value is greater than 0.05, so we cannot say that the coefficient is nonzero.")
        if len(signal_model.pvalues) < 4:
            pass
        elif signal_model.pvalues[3] < 0.05:
            st.success("The cubic term p-value is statistically significant")
        elif signal_model.pvalues[3] >= 0.05:
            st.error("The cubic term p-value is greater than 0.05, so we cannot say that the coefficient is nonzero.")
       

