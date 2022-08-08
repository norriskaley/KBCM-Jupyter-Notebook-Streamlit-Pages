import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
import plotly.express as px 

import sector_analysis
import home_sector_analysis

def main():
    st.set_page_config (layout="wide")
    with st.sidebar:
        view_selection = st.selectbox(
			'Select a view',
			('All Sectors', 'Specific Sector')
        )
    if view_selection == 'All Sectors':
        home_sector_analysis.main()
    elif view_selection == 'Specific Sector':
        sector_analysis.main()
    else:
        pass

if __name__ == '__main__':
    main()
    