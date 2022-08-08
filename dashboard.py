# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:57:54 2022

@author: NORRIKA
"""

import home
import streamlit as st
import signals
import signals_with_indicators
import properties
import nonlin_regression
# import signals_with_indictators_cols
import home_universal
import signals_universal
import properties_universal
import signals_with_indicators_universal

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

# pd.read_csv('BondSignalData.csv')
# data = pd.read_csv('BondSignalData.csv')

def main():
    st.set_page_config (layout="wide")
    with st.sidebar:
        view_selection = st.selectbox(
			'Select a view',
			('Home', 'Signals', 'Signals with Indicator Variables', 'Properties')
        )
    if view_selection == 'Home 2':
        home.main()
    elif view_selection == 'Signals 2':
        signals.main()
    elif view_selection == 'Signals with Indicator Variables 2':
        signals_with_indicators.main()
    elif view_selection == 'Properties 2':
        properties.main()
    # elif view_selection == 'cols':
    #     signals_with_indictators_cols.main()
    elif view_selection == 'Home':
        home_universal.main()
    elif view_selection == 'Signals':
        signals_universal.main()
    elif view_selection == 'Properties':
        properties_universal.main()
    elif view_selection == 'Signals with Indicator Variables':
        signals_with_indicators_universal.main()
    else:
        pass
    print(view_selection)
if __name__ == '__main__':
    main()
    