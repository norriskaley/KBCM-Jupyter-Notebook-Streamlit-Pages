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

    st.title("Creating an Indictator Variable")
    st.write("If you believe an indictator variable(s) would make the Linear Regression model more accurate, this page will allow you to add those for the signal of your choosing.")
    signal_selector = st.sidebar.selectbox("What signal would you like the indictator variable for?", ("signal3", "signal2", "signal1"))
    signal_add_indictator = st.sidebar.text_input("Enter the value at which you want the indictator variable", 1)
    filtered_data['signal_equal_indictator'] = filtered_data[signal_selector].apply(lambda x: 1 if x == float(signal_add_indictator) else 0)
    another_indictator = st.sidebar.radio("Would you like to create another indicator variable?", ("No", "Yes"))
    if another_indictator == "Yes":
        signal_add_indictator2 = st.sidebar.text_input("Enter the value at which you want another indictator variable", 1)
        filtered_data['signal_equal_indictator2'] = filtered_data[signal_selector].apply(lambda x: 1 if x == float(signal_add_indictator2) else 0)
        x = filtered_data[[signal_selector, 'signal_equal_indictator', 'signal_equal_indictator2']]
        y = filtered_data['move_1D']
 
        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()
        predictions = model.predict(x) 
        filtered_data['model_params'] = model.params[0]

        print_model = model.summary()
        
        #graph LinRrg
        fig2 = px.scatter(filtered_data, x=signal_selector, y='move_1D')
        fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
        fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
        fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
        st.write(fig5)
        st.write(print_model)
    else:
        if signal_selector == "signal3":
            x = filtered_data[[signal_selector, 'signal_equal_indictator']]
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model = sm.OLS(y, x).fit()
            predictions = model.predict(x) 
            filtered_data['model_params'] = model.params[0]

            print_model = model.summary()
            
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y='move_1D')
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif signal_selector == "signal2":
            x = filtered_data[[signal_selector, 'signal_equal_indictator']]
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model = sm.OLS(y, x).fit()
            predictions = model.predict(x) 
            filtered_data['model_params'] = model.params[0]

            print_model = model.summary()
            
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y='move_1D')
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        elif signal_selector == "signal1":
            x = filtered_data[[signal_selector, 'signal_equal_indictator']]
            y = filtered_data['move_1D']

            x = sm.add_constant(x)

            model = sm.OLS(y, x).fit()
            predictions = model.predict(x) 
            filtered_data['model_params'] = model.params[0]

            print_model = model.summary()
            
            #graph LinRrg
            fig2 = px.scatter(filtered_data, x=signal_selector, y='move_1D')
            fig3 = px.line(filtered_data, x=signal_selector, y=predictions, color_discrete_sequence = ['red'])
            fig4 = px.line(filtered_data, x=signal_selector, y='model_params', color_discrete_sequence = ['black'])
            fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
            st.write(fig5)
            st.write(print_model)
        else:
            pass
    error = rmse(y, predictions)
    st.write("The RMSE is",error)


# def main():
#     st.title("Creating an Indictator Variable")
#     st.write("If you believe an indictator variable(s) would make the Linear Regression model more accurate, this page will allow you to add those for the signal of your choosing.")
#     signal_add_indictator = st.sidebar.selectbox("What signal would you like the indictator variable for?", ("Signal 3", "Signal 2", "Signal 1"))
#     if signal_add_indictator == "Signal 3":
#         sig3indictator = st.sidebar.text_input("Enter the value at which you want the indictator variable")
#         data['sig3_indictator'] = data['signal3'].apply(lambda x: 1 if x == sig3indictator else 0)
#         signal_add_indictator2 = st.sidebar.radio("Would you like to create another indicator variable?", ("No", "Yes"))
#         if signal_add_indictator2 == "Yes":
#             sig3indictator2 = st.sidebar.text_input("Enter the value at which you want the second indictator variable")
#             data['sig3_indictator2'] = data['signal3'].apply(lambda x: 1 if x == sig3indictator2 else 0)
#             x = data[['signal3', 'sig3_indictator', 'sig3_indictator2']]
#             y = data['move_1D']

#             x = sm.add_constant(x)

#             model = sm.OLS(y, x).fit()
#             predictions = model.predict(x) 
#             data['model_params'] = model.params[0]

#             print_model = model.summary()
#             st.write(print_model)
#             #graph LinRrg
#             fig2 = px.scatter(data, x='signal3', y='move_1D')
#             fig3 = px.line(data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
#             fig4 = px.line(data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
#             fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#             st.write(fig5)
#         pass
#     #trying linReg
#         x = data[['signal3', 'sig3_indictator']]
#         y = data['move_1D']

#         x = sm.add_constant(x)

#         model = sm.OLS(y, x).fit()
#         predictions = model.predict(x) 
#         data['model_params'] = model.params[0]

#         print_model = model.summary()
#         st.write(print_model)
#         #graph LinRrg
#         fig2 = px.scatter(data, x='signal3', y='move_1D')
#         fig3 = px.line(data, x='signal3', y=predictions, color_discrete_sequence = ['red'])
#         fig4 = px.line(data, x='signal3', y='model_params', color_discrete_sequence = ['black'])
#         fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#         st.write(fig5)
#     elif signal_add_indictator == "Signal 2":
#         sig2indictator = st.sidebar.text_input("Enter the value at which you want the indictator variable")
#         data['sig2_indictator'] = data['signal2'].apply(lambda x: 1 if x == sig2indictator else 0)
#         signal_add_indictator2 = st.sidebar.radio("Would you like to create another indicator variable?", ("No", "Yes"))
#         if signal_add_indictator2 == "Yes":
#             sig2indictator2 = st.sidebar.text_input("Enter the value at which you want the second indictator variable")
#             data['sig2_indictator2'] = data['signal2'].apply(lambda x: 1 if x == sig2indictator2 else 0)
#         pass
#     #LinReg
#         x = data[['signal2', 'sig2_indictator']]
#         y = data['move_1D']

#         x = sm.add_constant(x)

#         model = sm.OLS(y, x).fit()
#         predictions = model.predict(x) 
#         data['model_params'] = model.params[0]

#         print_model = model.summary()
#         st.write(print_model)
#     #graph LinRrg
#         fig2 = px.scatter(data, x='signal2', y='move_1D')
#         fig3 = px.line(data, x='signal2', y=predictions, color_discrete_sequence = ['red'])
#         fig4 = px.line(data, x='signal2', y='model_params', color_discrete_sequence = ['black'])
#         fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#         st.write(fig5)
#     elif signal_add_indictator == "Signal 1":
#         sig1indictator = st.sidebar.text_input("Enter the value at which you want the indictator variable")
#         data['sig1_indictator'] = data['signal1'].apply(lambda x: 1 if x == sig1indictator else 0)
#         signal_add_indictator2 = st.sidebar.radio("Would you like to create another indicator variable?", ("No", "Yes"))
#         if signal_add_indictator2 == "Yes":
#             sig1indictator2 = st.sidebar.text_input("Enter the value at which you want the second indictator variable")
#             data['sig1_indictator2'] = data['signal1'].apply(lambda x: 1 if x == sig1indictator2 else 0)
#         pass
#     #LinReg
#         x = data[['signal1', 'sig1_indictator']]
#         y = data['move_1D']

#         x = sm.add_constant(x)

#         model = sm.OLS(y, x).fit()
#         predictions = model.predict(x) 
#         data['model_params'] = model.params[0]

#         print_model = model.summary()
#         st.write(print_model)
#     #graph LinRrg
#         fig2 = px.scatter(data, x='signal1', y='move_1D')
#         fig3 = px.line(data, x='signal1', y=predictions, color_discrete_sequence = ['red'])
#         fig4 = px.line(data, x='signal1', y='model_params', color_discrete_sequence = ['black'])
#         fig5 = go.Figure(data = fig2.data + fig3.data + fig4.data)
#         st.write(fig5)
#     else:
#         pass