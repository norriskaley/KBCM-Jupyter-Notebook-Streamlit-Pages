# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:24:34 2022

@author: NORRIKA
"""
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
    signal_selector = st.selectbox("Select Signal", ("Signal 3", "Signal 2", "Signal 1"))

    if signal_selector == "Signal 3":
        st.subheader("Distribution Plot of Signal3")
        #distplot
        fig1 = px.histogram(filtered_data, x='signal3')
        st.write(fig1)
        regression_selector = st.sidebar.radio("What type of regression would you like to look at?", ("Linear", "Quadratic", "Cubic"))
        if regression_selector == "Cubic":
            st.subheader("Cubic Regression Model")
            x = filtered_data['signal3']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal3 + I(signal3**2) + I(signal3**3)'
            signal_cubic = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_cubic_params'] = signal_cubic.params[0]

            predictions = signal_cubic.predict(x) 
            print_model = signal_cubic.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal3', y='move_1D')
            fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal3', y='signal_cubic_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif regression_selector == "Quadratic":
            st.subheader("Quadratic Regression Model")
            x = filtered_data['signal3']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal3 + I(signal3**2)'
            signal_model = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_model_params'] = signal_model.params[0]

            predictions = signal_model.predict(x) 
            print_model = signal_model.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal3', y='move_1D')
            fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal3', y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        else:
            move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
            st.subheader("Linear Regression Model")
            #LinReg
            if move_selector == "1 Day":
                
                x = filtered_data['signal3']
                y = filtered_data['move_1D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                #st.write(print_model)
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal3', y='move_1D')
                fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)
                
            elif move_selector == "3 days":
                x = filtered_data['signal3']
                y = filtered_data['move_3D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                #st.write(print_model)
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal3', y='move_3D')
                fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)
            else:
                pass

    if signal_selector == "Signal 2":
        st.subheader("Distribution Plot of Signal2")
        #distplot
        fig4 = px.histogram(filtered_data, x='signal2')
        st.write(fig4)

        regression_selector = st.sidebar.radio("what type of regression would you like to look at?", ("Linear", "Quadratic", "Cubic"))
        if regression_selector == "Cubic":
            st.subheader("Cubic Regression Model")
            x = filtered_data['signal2']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal2 + I(signal2**2) + I(signal2**3)'
            signal_cubic = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_cubic_params'] = signal_cubic.params[0]

            predictions = signal_cubic.predict(x) 
            print_model = signal_cubic.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal2', y='move_1D')
            fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal2', y='signal_cubic_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif regression_selector == "Quadratic":
            st.subheader("Quadratic Regression Model")
            x = filtered_data['signal2']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal2 + I(signal2**2)'
            signal_model = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_model_params'] = signal_model.params[0]

            predictions = signal_model.predict(x) 
            print_model = signal_model.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal2', y='move_1D')
            fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal2', y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        else:
            move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
            st.subheader("Linear Regression Model")
            #LinReg
            if move_selector == "1 Day":
                x = filtered_data['signal2']
                y = filtered_data['move_1D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                #st.write(print_model)
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal2', y='move_1D')
                fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal2', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)

            elif move_selector == "3 days":
                x = filtered_data['signal2']
                y = filtered_data['move_3D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                #st.write(print_model)
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal2', y='move_3D')
                fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal2', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)
            else:
                pass

    if signal_selector == "Signal 1":
        st.subheader("Distribution Plot of Signal1")
        #distplot
        fig7 = px.histogram(filtered_data, x='signal1')
        st.write(fig7)
        regression_selector = st.sidebar.radio("what type of regression would you like to look at?", ("Linear", "Quadratic", "Cubic"))
        if regression_selector == "Cubic":
            st.subheader("Cubic Regression Model")
            x = filtered_data['signal1']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal1 + I(signal1**2) + I(signal1**3)'
            signal_cubic = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_cubic_params'] = signal_cubic.params[0]

            predictions = signal_cubic.predict(x) 
            print_model = signal_cubic.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal1', y='move_1D')
            fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal1', y='signal_cubic_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif regression_selector == "Quadratic":
            st.subheader("Quadratic Regression Model")
            x = filtered_data['signal1']
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model= 'move_1D ~ signal1 + I(signal1**2)'
            signal_model = smf.ols(formula=model, data=filtered_data).fit()
            filtered_data['signal_model_params'] = signal_model.params[0]

            predictions = signal_model.predict(x) 
            print_model = signal_model.summary()
            
             #graph Rrg
            fig2 = px.scatter(filtered_data, x='signal1', y='move_1D')
            fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x='signal1', y='signal_model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        else:
            move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
            st.subheader("Linear Regression Model")
            #LinReg
            if move_selector == "1 Day":
                x = filtered_data['signal1']
                y = filtered_data['move_1D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal1', y='move_1D')
                fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal1', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)

            elif move_selector == "3 days":
                x = filtered_data['signal1']
                y = filtered_data['move_3D']

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()
                predictions = model.predict(x) 
                filtered_data['model_params'] = model.params[0]

                print_model = model.summary()
                
                #graph LinRrg
                fig2 = px.scatter(filtered_data, x='signal1', y='move_3D')
                fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
                fig4 = px.line(filtered_data, x='signal1', y='model_params', color_discrete_sequence = ['black'])
                fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
                st.write(fig5)
                st.write(print_model)
            else:
                pass
    else:
        pass 
        
        
        
        
        
        
        
#     #OLD  
#     st.title("Looking at the Different Signals")
#     signal_selector = st.selectbox("Select Signal", ("Signal 3", "Signal 2", "Signal 1"))

#     if signal_selector == "Signal 3":    
#         st.subheader("Distribution Plot of Signal3")
#         #distplot
#         fig1 = px.histogram(filtered_data, x='signal3')
#         st.write(fig1)
#         move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
#         st.subheader("Linear Regression Model")
#         #LinReg
#         if move_selector == "1 Day":
            
#             x = filtered_data['signal3']
#             y = filtered_data['move_1D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal3', y='move_1D')
#             fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)
            
#         elif move_selector == "3 days":
#             x = filtered_data['signal3']
#             y = filtered_data['move_3D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal3', y='move_3D')
#             fig3 = px.line(filtered_data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)
#         else:
#             pass
     
#     elif signal_selector == "Signal 2":
#         st.subheader("Distribution Plot of Signal2")
#         #distplot
#         fig4 = px.histogram(filtered_data, x='signal2')
#         st.write(fig4)
#         move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
#         st.subheader("Linear Regression Model")
#         #LinReg
#         if move_selector == "1 Day":
#             x = filtered_data['signal2']
#             y = filtered_data['move_1D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal2', y='move_1D')
#             fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal2', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)

#         elif move_selector == "3 days":
#             x = filtered_data['signal2']
#             y = filtered_data['move_3D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal2', y='move_3D')
#             fig3 = px.line(filtered_data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal2', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)
#         else:
#             pass

#     elif signal_selector == "Signal 1":
#         st.subheader("Distribution Plot of Signal1")
#         #distplot
#         fig7 = px.histogram(filtered_data, x='signal1')
#         st.write(fig7)

#         move_selector = st.selectbox("Movement After", ("1 Day", "3 days"))
#         st.subheader("Linear Regression Model")
#         #LinReg
#         if move_selector == "1 Day":
#             x = filtered_data['signal1']
#             y = filtered_data['move_1D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal1', y='move_1D')
#             fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal1', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)

#         elif move_selector == "3 days":
#             x = filtered_data['signal1']
#             y = filtered_data['move_3D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             filtered_data['model_params'] = model.params[0]

#             print_model = model.summary()
#             #st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(filtered_data, x='signal1', y='move_3D')
#             fig3 = px.line(filtered_data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(filtered_data, x='signal1', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)
#         else:
#             pass
#     else:
#         pass
    
#     error = rmse(y, predictions)
#     st.write("The RMSE is",error)
#     st.write("The red line shows the predicted movement.")
# #    st.sidebar.markdown("Show Analysis for Movement after:")
# #    st.sidebar.checkbox("1 Day", True, key=1)
# #    st.sidebar.checkbox("3 Days", True, key=1)
