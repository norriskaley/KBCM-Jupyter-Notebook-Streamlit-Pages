#!/usr/bin/env python
# coding: utf-8

# # Loading and Setting up Data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import os
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


os.getcwd()
os.chdir(r'C:\Users\NORRIKA\OneDrive - Key Corp\Documents\Project')


# In[3]:


pd.read_csv('BondSignalData.csv')
data = pd.read_csv('BondSignalData.csv')


# In[4]:


data['rqdatetime'] = pd.to_datetime(data.request_time)
data['rfqdate']=data['rqdatetime'].dt.date

data['year']=pd.to_datetime(data.request_time).dt.year
data['month']=data.rqdatetime.dt.month
data['day']=pd.to_datetime(data.request_time).dt.day
data['hour']=pd.to_datetime(data.request_time).dt.hour
data['day']= data['day'].apply(lambda x: '{0:0>2}'.format(x)) #turning day to two digits so 1-04 comes before 1-11
# data['month_day']= (data['month'].astype(str) + "-" + data['day'].astype(str)) #putting month & day together
data['month'] = data['month'].replace(4, 3) #as only have one day of data in April

data['bondlength'] = data.mat_bucket.str.strip("B")
data['bondlength']=data.bondlength.astype(int)

#for regression
data['sig3equal1'] = data['signal3'].apply(lambda x: 1 if x >= 1 else 0)
data['sig3equalneg1'] = data['signal3'].apply(lambda x: 1 if x <= -1 else 0)

#Creating Buckets
data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])


# # Exploring relationships between variables

# In[5]:


sns.relplot(x="industrySector", y="liq_score", hue="binnedmat_bucket", palette="ch:r=-.5,l=.75", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# The Energy and Financial sectors seem to be majority of the 2-3 year maturity bonds.

# ### Maturity vs Liquidity 

# In[24]:


sns.boxplot(x="bondlength", y="liq_score", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[7]:


data['liq_score'].corr(data['bondlength'])


# In[115]:


pd.crosstab(data['binnedliq_score'], data['binnedmat_bucket']).plot(kind='pie', subplots=True)
plt.rcParams["figure.figsize"]=50,50


# a lower rank liquidity score (aka a more liquid bond) seems to correspond to bonds with a longer maturity. (see above) Liquidity bins 1 & 2 make up majority of the last two pies

# ### Sector vs Liquidity 

# In[20]:


sns.boxplot(x="industrySector", y="liq_score", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# The most liquid bonds correspond to the sector Consumer Non-cyclcial with Basic Materials following next.

# In[200]:


#INDUSTRY SECTOR
industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for sector in industrySector:
       print(sector, data['liq_score'].corr(data['industrySector']==sector))


# In[187]:


#LIQUIDITY SCORE --labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0]
#5=(719.0, 1065.0] 6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]


# In[82]:


pd.crosstab(data['industrySector'], data['binnedliq_score']).plot(kind='pie', subplots=True)
plt.rcParams["figure.figsize"]=75,75


# Consumer Non-cyclical and Basic Materials I later find to have the largest slope coeff and greatest R-squared values. These two sectors make up majority of the most liquid bonds which I also found to have the greatest signal slope and R-squared values. 

# ### Sector vs Quantity

# In[28]:


sns.boxplot(x="industrySector", y="quantity_bonds", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# Quantity ranges for all sectors but the graph is most dense for all sectors at a lower quantity and is more sparse at higher quantity amounts.

# In[201]:


#INDUSTRY SECTOR
industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for sector in industrySector:
       print(sector, data['quantity_bonds'].corr(data['industrySector']==sector))


# In[202]:


pd.crosstab(data['industrySector'], data['binnedquantity_bonds']).plot(kind='pie', subplots=True)
plt.rcParams["figure.figsize"]=50,50


# ### Maturity vs Quantity

# In[26]:


sns.boxplot(x="bondlength", y="quantity_bonds", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[199]:


data['bondlength'].corr(data['quantity_bonds'])


# No relevant relationship; all maturity bonds touch about the same quantity amounts

# ### Sector vs Maturity 

# In[15]:


# Cross tabulation between industrySector and mat_bucket
CrosstabResult=pd.crosstab(index=data['industrySector'],columns=data['mat_bucket'])
print(CrosstabResult)
 
# importing the required function
from scipy.stats import chi2_contingency
 
# Performing Chi-sq test
ChiSqResult = chi2_contingency(CrosstabResult)
 
# P-Value is the Probability of H0 being True
# If P-Value>0.05 then only we Accept the assumption(H0)
 
print('The P-Value of the ChiSq Test is:', ChiSqResult[1])
#0-0.5 = variables related; greater than 0.05 = not correlated


# this cross tabulation shows how many RFQs from each sector are a specific maturity length

# In[54]:


sns.catplot(x="industrySector", y="bondlength", data=data)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[9]:


pd.crosstab(data['industrySector'], data['binnedmat_bucket']).plot(kind='pie', subplots=True)
plt.rcParams["figure.figsize"]=50,50


# Financial makes up over half of the 2-3 year bonds which I later find Financial sector bonds to have the worst R2 values and so do 2-3 year maturities. Technology makes up most of 5 year bonds and Consumer Non-cyclical makes up majority of the longer maturity bonds (10 & 30 years)

# # Exploring with Graphs

# I started playing around with some graphs to get used to Python and I fell across these two below. From here, I started to see a positive correlation between signal3 and move_1d data and decided to dive a little deeper into their relationship.

# In[20]:


sns.pairplot(data[['cusip','industrySector','ticker','bondlength', 'quantity_bonds','liq_score','move_1D','move_3D','signal1', 'signal2', 'signal3','rqdatetime','month']])


# In[10]:


mask = np.triu(np.ones_like(data[['cusip','industrySector','ticker','bondlength', 'quantity_bonds','liq_score','move_1D','move_3D','signal1', 'signal2', 'signal3','rqdatetime','month']].corr(), dtype=bool))
np.fill_diagonal(mask, False)
matrix = data[['cusip','industrySector','ticker','bondlength', 'quantity_bonds','liq_score','move_1D','move_3D','signal1', 'signal2', 'signal3','rqdatetime','month']].corr().round(3)
sns.heatmap(matrix, mask=mask, annot=True)
plt.show()


# # SIG3 MOVE_1D

# I liked how this graph combined the two variables to see their rather positive relationship but also showed their indiviudal distribution plots below. I was playing around at first and came across this graph by accident which plots the variables on the wrong axes but it gave way to further exploration in these two variables so I decided to keep this graph in my notebook.

# In[35]:


fig, ax = plt.subplots()
sns.jointplot(x=data['move_1D'], y= data['signal3'], kind ='kde', ax=ax)
ax.set_xlim(-1,1)
ax.set_ylabel('signal3')
ax.set_xlabel('move_1D')
plt.show()


# In[135]:


sns.jointplot(x=data['signal3'], y= data['move_1D'])
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal3')
plt.show()


# The illustration above gave light to the interesting distribution of signal3 so the following graph enlarges that individual distribution image.

# In[36]:


sns.distplot(data['signal3'])


# I started binning variables starting with signal3 as the graph above showcased an interesting distribution with the ends containing a lot of data and the middle resembling a normal distribution. These bins then I could use to look at the move_1D in that specific bucket.

# In[85]:


#SIGNAL3   1=[-1.25, -0.75)  2=[-0.75, -0.75)  3=[0.75, 1.25]
bins = [-1.25, -0.75, 0.75, 1.25]
data['binnedsig3'] = pd.cut(data['signal3'], bins, labels = [1,2,3])
data.head()


# In[79]:


sns.boxplot(x=data['binnedsig3'], y=data['move_1D'])


# the positive correlation between these two varaiables is certainly present above.  

# In[80]:


data['move_1D'].corr(data['binnedsig3']==1)


# In[81]:


data['move_1D'].corr(data['binnedsig3']==2)


# In[39]:


data['move_1D'].corr(data['binnedsig3']==3)


# ## Regression

# In[411]:


x = data[['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[86]:


sns.lineplot(x = data['signal3'], y=predictions)
sns.lineplot(x = data['signal3'], y=model.params[0], color = 'red')
plt.scatter(data['signal3'],data['move_1D'], color = 'black')


# In[76]:


from statsmodels.tools.eval_measures import rmse
# calc rmse
error = rmse(y, predictions)


# In[77]:


error


# In[65]:


jan_feb = data[data.month<3][['signal3', 'sig3equal1', 'sig3equalneg1', 'move_1D']]
march = data[data.month==3][['signal3', 'sig3equal1', 'sig3equalneg1', 'move_1D']]


# In[68]:


x_train = jan_feb[['signal3', 'sig3equal1', 'sig3equalneg1']]
y_train = jan_feb['move_1D']
x_test = march[['signal3', 'sig3equal1', 'sig3equalneg1']]
y_test = march['move_1D']


# In[70]:


x_train = sm.add_constant(x_train)

model = sm.OLS(y_train, x_train).fit()
predictions_train = model.predict(x_train)
# predictions_test = model.predict(x_test)

print_model = model.summary()
print(print_model)


# In[72]:


predictions_test = -0.0045 + 0.7931*(x_test['signal3']) + 0.6433*(x_test['sig3equal1']) - 0.6049*(x_test['sig3equalneg1'])


# In[73]:


predictions_test


# In[74]:


predictions_train


# In[78]:


rmse_train = rmse(y_train, predictions_train)
rmse_test =  rmse(y_test, predictions_test)


# In[79]:


rmse_train


# In[80]:


rmse_test


# # Exploring predictors/features that help the signal

# Next, I transitioned into looking at features/characteristics that may help signal3 predict the move_1D data. Liquidity score and bond quantity had right skewed distributions so I started exploring those features and buckets first as shown below. 

# ### Liquidity Score 

# In[117]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
labels=[1,2,3,4,5,6,7,8]
for i in labels:
    print(i, data[data['binnedliq_score']== i]['move_1D'].corr(data[data['binnedliq_score']== i]['signal3']))


# The bonds/cusips that are the most liquid (traded most often, lower score) have better signal3 data to help predict move_1D data. (83.5% at most liquid, 76.4% in middle, 73.6% at least liquid cusips)

# In[40]:


plt.hist(data['liq_score'], bins=25, density=True, alpha=0.6, color='b')


# In[7]:


#LIQUIDITY SCORE --labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0] 5=(719.0, 1065.0] 6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]
data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
data.groupby(['binnedliq_score']).count()


# In[8]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
labels=[1,2,3,4,5,6,7,8]
for i in labels:
    x = data[data['binnedliq_score']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedliq_score']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[54]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal3 and binnedliq_score along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + binnedliq_score + signal3*binnedliq_score + sig3equal1*binnedliq_score + sig3equalneg1*binnedliq_score'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# In[55]:


print(data.loc[567][['signal3','binnedliq_score','sig3equal1','sig3equalneg1']])

#print(intercept + binnedliq_score[] + data[signal3]*(signal3 - signal3*binnedliq_score[]))
print(0.0048 + -0.0151 + -0.519567*(0.8923 - 0.0757))

print(model.predict(data)[567])


# ### Quantity of Bonds 

# In[18]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
labels=['1-12', '12-50', '50-210', '210-600', '600-3000']
for i in labels:
    print(i, data[data['binnedquantity_bonds']== i]['move_1D'].corr(data[data['binnedquantity_bonds']== i]['signal3']))


# After looking at the regressions of the different buckets, I did not find this variable to have suffifient data to prove that it helps signal3 predict the move_1D data.

# In[57]:


plt.hist(data['quantity_bonds'])


# In[75]:


#BOND QUANTITY
data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
data.groupby(['binnedquantity_bonds']).count()


# In[19]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
labels=['1-12', '12-50', '50-210', '210-600', '600-3000']
for i in labels:
    x = data[data['binnedquantity_bonds']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedquantity_bonds']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[48]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal3 and binnedquantity_bonds along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + binnedquantity_bonds + signal3*binnedquantity_bonds + sig3equal1*binnedquantity_bonds + sig3equalneg1*binnedquantity_bonds'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Maturity of Bond 

# In[13]:


#BOND MATURITY
bondlength= [2,3,5,10,30]
for bond in bondlength:
    print(bond, data[data['bondlength']== bond]['move_1D'].corr(data[data['bondlength']== bond]['signal3']))


# I did however find data highlighting the importance of the maturity of the bond. Bond maturity is most helpful to signal3 in predicting move_1D movement with a 30 year bond. (most helpful with greater maturity bonds than ones with a lower maturity age)

# In[11]:


#MATURITY
data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
data.groupby(['binnedmat_bucket']).count()


# In[20]:


data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
labels=['2-3 years', '5 years', '10 years', '30 years']
for i in labels:
    x = data[data['binnedmat_bucket']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedmat_bucket']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[67]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal3 and binnedmat_bucket along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + binnedmat_bucket + signal3*binnedmat_bucket + sig3equal1*binnedmat_bucket + sig3equalneg1*binnedmat_bucket'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Industry Sector 

# In[12]:


#INDUSTRY SECTOR
industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for sector in industrySector:
       print(sector, data[data['industrySector']== sector]['move_1D'].corr(data[data['industrySector']== sector]['signal3']))


# Some sectors seem to be better at aiding signal3 in predicting move_1D data. Financial Sector bonds are definitely the worst group. 

# In[21]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for i in industrySector:
    x = data[data['industrySector']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['industrySector']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[56]:


data['sector_category']= data.industrySector.astype('category')


# In[57]:


import statsmodels.formula.api as smf #Basic Materials is first group
#move-1d data interacted with signal3 and sector_category along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + sector_category + signal3*sector_category + sig3equal1*sector_category + sig3equalneg1*sector_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Ticker 

# I kept this in as I thought it was interesting how some tickers are more helpful than others but hard to really take this into account as some tickers only have 1-2 RFQs. Therefore there is not a sufficient amount of data to be able to properly interpret their relationship. If I had more time, I would have made a bucket of tickers with less than 300 RFQs together, left the RFQs like MSFT and CVS with 1000+ shares and then compared signal3 and move_1D with those buckets. 

# In[124]:


data.groupby('ticker').count()


# In[22]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for i in ticker:
    x = data[data['ticker']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['ticker']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[210]:


import statsmodels.formula.api as smf #Basic Materials is first group
#move-1d data interacted with signal3 and ticker along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + ticker + signal3*ticker + sig3equal1*ticker + sig3equalneg1*ticker'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Month

# Looked at if the month had any significance in helping signal3 predict move_1D data. no real significance found. all signal3 coeff are relatively the same each month.

# In[26]:


data.month.sort_values().unique()


# In[64]:


month=[1,2,3]
for i in month:
    x = data[data['month']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['month']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[65]:


data['month_category']=data.month.astype('category')


# In[66]:


import statsmodels.formula.api as smf #Basic Materials is first group
#move-1d data interacted with signal3 and month_category along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_1D ~ signal3 + month_category + signal3*month_category + sig3equal1*month_category + sig3equalneg1*month_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# # SIG3 MOVE_3D

# In[86]:


#SIGNAL3   1=[-1.25, -0.75)  2=[-0.75, -0.75)  3=[0.75, 1.25]
bins = [-1.25, -0.75, 0.75, 1.25]
data['binnedsig3'] = pd.cut(data['signal3'], bins, labels = [1,2,3])
data.head()


# In[9]:


sns.boxplot(x=data['binnedsig3'], y=data['move_3D'])


# the positive correlation between these two varaiables is certainly present above. as seen below, the correlations are less than with move_1D data. those were around -0.60 and 0.60 instead of -0.40 and 0.40 below

# In[10]:


data['move_3D'].corr(data['binnedsig3']==1)


# In[11]:


data['move_3D'].corr(data['binnedsig3']==2)


# In[12]:


data['move_3D'].corr(data['binnedsig3']==3)


# ## Regression

# Our regression model has greater variance between (-1,1) which makes it harder for our model and signal3 to predict the move_3D movement thus why the R-squared value is less; only 30.9% of move_3D data is explained by signal3 whereas 76.6% of data was explained for move_1D.

# In[24]:


x = data[['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data['move_3D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[17]:


sns.lineplot(x = data['signal3'], y=predictions)
sns.lineplot(x = data['signal3'], y=model.params[0], color = 'red')
plt.scatter(data['signal3'],data['move_3D'], color = 'black')


# # Exploring predictors/features that help the signal

# ### Liquidity Score 

# R-squared values do not increase/decrease in order. signal3 coeffs decrease from bin1-3, increase from bin4-6 and decrease again

# In[14]:


#LIQUIDITY SCORE --labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0] 5=(719.0, 1065.0] 
#6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]
data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
labels=[1,2,3,4,5,6,7,8]
for i in labels:
    x = data[data['binnedliq_score']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedliq_score']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[15]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and binnedliq_score along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + binnedliq_score + signal3*binnedliq_score + sig3equal1*binnedliq_score + sig3equalneg1*binnedliq_score'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# In[16]:


print(data.loc[398][['signal3','binnedliq_score','sig3equal1','sig3equalneg1']])

#print(intercept + binnedliq_score[] + data[signal3]*(signal3 - signal3*binnedliq_score[]))
print(-0.2626 + 0.0859 - 0.133603*(1.6460 - 0.5722))

print(model.predict(data)[398])


# In[21]:


print(data.loc[201][['signal3','binnedliq_score','sig3equal1','sig3equalneg1']])

#print(intercept + binnedliq_score[] + data[signal3]*(signal3 - signal3*binnedliq_score[]) + data[sig3equal1]*(sig3equal1 - sig3equal1*bineedliq_score[]))
print(-0.2626 + 0.1300 + 1.0*(1.6460 - 0.8303) + 1*(0.6899 + 0.1527))

print(model.predict(data)[201])


# ### Quantity of Bonds 

# coeff and R-squared terms do not range much from bin to bin. R-squared values do increase slightly with bin number/quantity increasing but nothing great to note.

# In[24]:


#BOND QUANTITY
data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
data.groupby(['binnedquantity_bonds']).count()


# In[25]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
labels=['1-12', '12-50', '50-210', '210-600', '600-3000']
for i in labels:
    x = data[data['binnedquantity_bonds']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedquantity_bonds']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[26]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and binnedquantity_bonds along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + binnedquantity_bonds + signal3*binnedquantity_bonds  + sig3equal1*binnedquantity_bonds  + sig3equalneg1*binnedquantity_bonds'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# a lot of the p-values in the interaction model are not statistically significant

# ### Maturity of Bond 

# The maturity of the bond does not help signal3 as well with move_3D as it did with move_1D. (R-squared) 30.9% at 2-3 years, 22.8% at 5 years, 30.6% at 10 years, and 37.1% at 30 years. (P-values) however are all statistically significant 
# 

# In[27]:


#MATURITY
data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
data.groupby(['binnedmat_bucket']).count()


# In[30]:


data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
labels=['2-3 years', '5 years', '10 years', '30 years']
for i in labels:
    x = data[data['binnedmat_bucket']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['binnedmat_bucket']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[71]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and binnedmat_bucket along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + binnedmat_bucket + signal3*binnedmat_bucket + sig3equal1*binnedmat_bucket + sig3equalneg1*binnedmat_bucket'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Industry Sector 

# Some sectors seem to be better at aiding signal3 in predicting move_3D data. Technology Sector bonds are the worst group. 

# In[35]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for i in industrySector:
    x = data[data['industrySector']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['industrySector']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[11]:


data['sector_category']= data.industrySector.astype('category')


# In[72]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and sector_category along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + sector_category + signal3*sector_category + sig3equal1*sector_category + sig3equalneg1*sector_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Ticker 

# In[36]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for i in ticker:
    x = data[data['ticker']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['ticker']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[175]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and ticker along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + ticker + signal3*ticker + sig3equal1*ticker + sig3equalneg1*ticker'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# # Month

# Looked at if the month had any significance in helping signal3 predict move_3D data. month=3 seems to have the best data

# In[26]:


data.month.sort_values().unique()


# In[44]:


month=[1,2,3]
for i in month:
    x = data[data['month']==i][['signal3', 'sig3equal1', 'sig3equalneg1']]
    y = data[data['month']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal3','sig3equal1', 'sig3equalneg1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[24]:


data['month_category']=data.month.astype('category')


# In[70]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal3 and month_category along with sig3, sig3equal1, sig3equalneg1 in each bin
formula = 'move_3D ~ signal3 + month_category + signal3*month_category + sig3equal1*month_category + sig3equalneg1*month_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# # SIG1 MOVE_1D

# ### Signal1 

# In[155]:


sns.distplot(data['signal1'])


# In[154]:


sns.jointplot(x=data['signal1'], y= data['move_1D'])
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal1')
plt.show()


# In[328]:


#SIGNAL1--labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0] 5=(719.0, 1065.0] 6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]
data['binnedsig1']= pd.qcut(data['signal1'], q=5, labels=[1,2,3,4,5])
data.groupby(['binnedsig1']).count()


# In[331]:


sns.boxplot(x=data['binnedsig1'], y=data['move_1D'])


# the correlation is somewhat positive between these two variables but is quite close to zero. the correlation values bounce above and below zero as shown below.

# In[332]:


data['move_1D'].corr(data['binnedsig1']==1)


# In[337]:


data['move_1D'].corr(data['binnedsig1']==2)


# In[334]:


data['move_1D'].corr(data['binnedsig1']==3)


# In[338]:


data['move_1D'].corr(data['binnedsig1']==4)


# In[335]:


data['move_1D'].corr(data['binnedsig1']==5)


# ## Regression

# In[85]:


x = data['signal1']
y = data['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[92]:


sns.lineplot(x = data['signal1'], y=predictions)
sns.lineplot(x = data['signal1'], y=model.params[0], color = 'red')
plt.scatter(data['signal1'],data['move_1D'], color = 'black')


# In[87]:


from statsmodels.tools.eval_measures import rmse

# calc rmse
error_1 = rmse(y, predictions)


# In[88]:


error_1


# quadratic

# In[425]:


x = data['signal1']
y = data['move_1D']

x = sm.add_constant(x)

model= 'move_1D ~ signal1 + I(signal1**2)'
signal1_model = smf.ols(formula=model, data=data).fit()

predictions = signal1_model.predict(x) 
print_model = signal1_model.summary()
print(print_model)


# In[429]:


sns.lineplot(x = data['signal1'], y=predictions)
sns.lineplot(x = data['signal1'], y=signal1_model.params[0], color = 'red')
plt.scatter(data['signal1'],data['move_1D'], color = 'black')


# In[426]:


from statsmodels.tools.eval_measures import rmse

# calc rmse
error_2 = rmse(y, predictions)


# In[427]:


error_2


# cubic

# In[431]:


x = data['signal1']
y = data['move_1D']

x = sm.add_constant(x)

model= 'move_1D ~ signal1 + I(signal1**2) + I(signal1**3)'
signal1_cubic = smf.ols(formula=model, data=data).fit()

predictions = signal1_cubic.predict(x) 
print_model = signal1_cubic.summary()
print(print_model)


# In[432]:


sns.lineplot(x = data['signal1'], y=predictions)
sns.lineplot(x = data['signal1'], y=signal1_cubic.params[0], color = 'red')
plt.scatter(data['signal1'],data['move_1D'], color = 'black')


# In[433]:


from statsmodels.tools.eval_measures import rmse

error_3 = rmse(y, predictions)


# In[434]:


error_3


# # Exploring predictors/features that help the signal

# ### Liquidity Score

# In[286]:


#LIQUIDITY SCORE --labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0] 5=(719.0, 1065.0] 6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]
data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
data.groupby(['binnedliq_score']).count()


# In[146]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
labels=[1,2,3,4,5,6,7,8]
for i in labels:
    x = data[data['binnedliq_score']==i]['signal1']
    y = data[data['binnedliq_score']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[310]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and binnedliq_score along with sig1 in each bin
formula = 'move_1D ~ signal1 + binnedliq_score + signal1*binnedliq_score'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# In[313]:


print(data.loc[567][['signal1','binnedliq_score']])

#print(intercept + binnedliq_score[] + data[signal1]*(signal1 - signal1*binnedliq_score[]))
print(-0.1332 + 0.0808 + -0.65894*(0.7622 - 0.4915))

print(model.predict(data)[567])


# ### Quantity of Bonds 

# In[293]:


#BOND QUANTITY
data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
data.groupby(['binnedquantity_bonds']).count()


# In[294]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
labels=['1-12', '12-50', '50-210', '210-600', '600-3000']
for i in labels:
    x = data[data['binnedquantity_bonds']==i][['signal1']]
    y = data[data['binnedquantity_bonds']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[147]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and binnedquantity_bonds along with sig1 in each bin
formula = 'move_1D ~ signal1 + binnedquantity_bonds + signal1*binnedquantity_bonds'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Maturity of Bond 

# In[295]:


#MATURITY
data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
data.groupby(['binnedmat_bucket']).count()


# In[296]:


data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
labels=['2-3 years', '5 years', '10 years', '30 years']
for i in labels:
    x = data[data['binnedmat_bucket']==i][['signal1']]
    y = data[data['binnedmat_bucket']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[148]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and binnedmat_bucket along with sig1 in each bin
formula = 'move_1D ~ signal1 + binnedmat_bucket + signal1*binnedmat_bucket'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Industry Sector 

# Some sectors seem to be better at aiding signal1 in predicting move_1D data, specifically Basic Materials and Consumer Non-cyclical.

# In[297]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for i in industrySector:
    x = data[data['industrySector']==i][['signal1']]
    y = data[data['industrySector']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[149]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and sector_category along with sig1 in each bin
formula = 'move_1D ~ signal1 + sector_category + signal1*sector_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Ticker 

# In[298]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for i in ticker:
    x = data[data['ticker']==i][['signal1']]
    y = data[data['ticker']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[150]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and ticker along with sig1 in each bin
formula = 'move_1D ~ signal1 + ticker + signal1*ticker'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Month

# Looked at if the month had any significance in helping signal1 predict move_1D data. no real significance found. best in March but not by much.

# In[26]:


data.month.sort_values().unique()


# In[300]:


month=[1,2,3]
for i in month:
    x = data[data['month']==i][['signal1']]
    y = data[data['month']==i]['move_1D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[151]:


import statsmodels.formula.api as smf
#move-1d data interacted with signal1 and month_category along with sig1 in each bin
formula = 'move_1D ~ signal1 + month_category + signal1*month_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# # SIG1 MOVE_3D

# ## Regression

# In[31]:


x = data['signal1']
y = data['move_3D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[32]:


sns.lineplot(x = data['signal1'], y=predictions)
sns.lineplot(x = data['signal1'], y=model.params[0], color = 'red')
plt.scatter(data['signal1'],data['move_3D'], color = 'black')


# In[33]:


from statsmodels.tools.eval_measures import rmse

# calc rmse
error_4 = rmse(y, predictions)


# In[34]:


error_4


# cubic

# In[26]:


x = data['signal1']
y = data['move_3D']

x = sm.add_constant(x)

model= 'move_3D ~ signal1 + I(signal1**2) + I(signal1**3)'
signal1_cubic3D = smf.ols(formula=model, data=data).fit()

predictions = signal1_cubic3D.predict(x) 
print_model = signal1_cubic3D.summary()
print(print_model)


# In[28]:


sns.lineplot(x = data['signal1'], y=predictions)
sns.lineplot(x = data['signal1'], y=signal1_cubic3D.params[0], color = 'red')
plt.scatter(data['signal1'],data['move_3D'], color = 'black')


# In[29]:


from statsmodels.tools.eval_measures import rmse

error_5 = rmse(y, predictions)


# In[30]:


error_5


# # Exploring predictors/features that help the signal

# ### Liquidity Score

# In[286]:


#LIQUIDITY SCORE --labels: 1=(2.999, 12.0] 2=(12.0, 182.0] 3=(182.0, 399.0] 4=(399.0, 719.0] 5=(719.0, 1065.0] 6=(1065.0, 1394.0] 7=(1394.0, 2087.0] 8=(2087.0, 3000.0]
data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
data.groupby(['binnedliq_score']).count()


# In[302]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])
labels=[1,2,3,4,5,6,7,8]
for i in labels:
    x = data[data['binnedliq_score']==i]['signal1']
    y = data[data['binnedliq_score']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[153]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and binnedliq_score along with sig1 in each bin
formula = 'move_3D ~ signal1 + binnedliq_score + signal1*binnedliq_score'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# In[304]:


print(data.loc[567][['signal1','binnedliq_score']])

#print(intercept + binnedliq_score[] + data[signal1]*(signal1 - signal1*binnedliq_score[]))
print(-0.3593 + 0.1225 + -0.65894*(0.6018 - 0.3875))

print(model.predict(data)[567])


# ### Quantity of Bonds 

# In[293]:


#BOND QUANTITY
data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
data.groupby(['binnedquantity_bonds']).count()


# In[314]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])
labels=['1-12', '12-50', '50-210', '210-600', '600-3000']
for i in labels:
    x = data[data['binnedquantity_bonds']==i][['signal1']]
    y = data[data['binnedquantity_bonds']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[154]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and binnedquantity_bonds along with sig1 in each bin
formula = 'move_3D ~ signal1 + binnedquantity_bonds + signal1*binnedquantity_bonds'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Maturity of Bond 

# In[295]:


#MATURITY
data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
data.groupby(['binnedmat_bucket']).count()


# In[315]:


data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])
labels=['2-3 years', '5 years', '10 years', '30 years']
for i in labels:
    x = data[data['binnedmat_bucket']==i][['signal1']]
    y = data[data['binnedmat_bucket']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print('bin', i)
    print(sub_df)
    print('-------------')  


# In[155]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and binnedmat_bucket along with sig1 in each bin
formula = 'move_3D ~ signal1 + binnedmat_bucket + signal1*binnedmat_bucket'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Industry Sector 

# Sectors do not seem to matter much in helping signal1 explain move_3D data.

# In[316]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for i in industrySector:
    x = data[data['industrySector']==i][['signal1']]
    y = data[data['industrySector']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[156]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and sector_category along with sig1 in each bin
formula = 'move_3D ~ signal1 + sector_category + signal1*sector_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Ticker 

# In[317]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for i in ticker:
    x = data[data['ticker']==i][['signal1']]
    y = data[data['ticker']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[157]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and ticker along with sig1 in each bin
formula = 'move_3D ~ signal1 + ticker + signal1*ticker'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# ### Month

# Looked at if the month had any significance in helping signal1 predict move_1D data. best month = month 3

# In[26]:


data.month.sort_values().unique()


# In[318]:


month=[1,2,3]
for i in month:
    x = data[data['month']==i][['signal1']]
    y = data[data['month']==i]['move_3D']

    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 

    
    variable_coefs = model.params
    confidence_interval= model.conf_int(alpha=0.05)
    p_values= model.pvalues
    r_squared= model.rsquared
    
    sub_df = pd.DataFrame()
    sub_df['features'] = ['const', 'signal1']
    sub_df.set_index('features', inplace=True)
    sub_df['coef'] = list(variable_coefs)
    sub_df['conf_a'] = list(confidence_interval[0])
    sub_df['conf_b'] = list(confidence_interval[1])
    sub_df['p_values'] = list(p_values)
    sub_df['r_squared'] = r_squared
    
    print(i)
    print(sub_df)
    print('-------------')  


# In[158]:


import statsmodels.formula.api as smf
#move-3d data interacted with signal1 and month_category along with sig1 in each bin
formula = 'move_3D ~ signal1 + month_category + signal1*month_category'

model = smf.ols(formula = formula, data = data).fit()
model.summary()


# # SIG2 MOVE_1D

# ## Signal2

# In[163]:


sns.distplot(data['signal2'])


# In[161]:


sns.jointplot(x=data['signal2'], y= data['move_1D'])
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal2')
plt.show()


# In[202]:


#SIGNAL2--labels: 1=(-1.001, -0.594] 2=(-0.594, -0.207] 3=(-0.207, 0.195] 4=(0.195, 0.596] 5=(0.596, 1.0]
data['binnedsig2']= pd.qcut(data['signal2'], q=5, labels=[1,2,3,4,5])
data.groupby(['binnedsig2']).count()


# In[203]:


sns.boxplot(x=data['binnedsig2'], y=data['move_1D'])


# the lack of correlation between these two variables is evident with all the correlation values bouncing above and below 0.

# In[204]:


data['move_1D'].corr(data['binnedsig2']==1)


# In[205]:


data['move_1D'].corr(data['binnedsig2']==2)


# In[206]:


data['move_1D'].corr(data['binnedsig2']==3)


# In[207]:


data['move_1D'].corr(data['binnedsig2']==4)


# In[208]:


data['move_1D'].corr(data['binnedsig2']==5)


# ## Regression

# In[214]:


x = data['signal2']
y = data['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[215]:


sns.lineplot(x = data['signal2'], y=predictions)
sns.lineplot(x = data['signal2'], y=model.params[0], color = 'red')
plt.scatter(data['signal2'],data['move_1D'], color = 'black')


# what's happening with signal2 is too noisy to detect with the human eye and graphing. going to use tree models as those are better at finding relationships that are nonlinear.

# ## Trees

# In[38]:


sig2tree = DecisionTreeRegressor(max_depth=3)
x = data[['signal2', 'liq_score', 'bondlength', 'quantity_bonds']]
y = data['move_1D']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.20)


# In[39]:


ytest


# In[114]:


x


# In[115]:


sig2tree.fit(x,y)


# In[116]:


y_pred = sig2tree.predict(x)


# In[117]:


mse = mean_squared_error(y, y_pred)
rmse_tree = math.sqrt(mse)
rmse_tree


# In[206]:


sig2tree.fit(xtrain, ytrain)

ypred = sig2tree.predict(xtest)


# In[207]:


mse = mean_squared_error(ytest, ypred)
rmse_tree2 = math.sqrt(mse)
rmse_tree2


# X[0]= signal2  X[1]= liq_score X[2]= bondlength X[3]= quantity_bonds

# In[120]:


tree.plot_tree(sig2tree)
plt.rcParams["figure.figsize"]=15,15
plt.show()


# if guess zero for signal2, the rmse is the same as if i used my signal2 tree. so using the tree and signal2 is no better than just guessing zero everytime. shows the insignificance of tree.

# In[35]:


data['zero']=0


# In[36]:


data.zero


# In[40]:


mse = mean_squared_error(ytest, data.zero[0:2500])
rmse_zero = math.sqrt(mse)
rmse_zero


# # SIG2 MOVE_3D

# ## Regression

# In[97]:


x = data['signal2']
y = data['move_3D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[229]:


sns.lineplot(x = data['signal2'], y=predictions)
sns.lineplot(x = data['signal2'], y=model.params[0], color = 'red')
plt.scatter(data['signal2'],data['move_3D'], color = 'black')


# ## Trees

# The RMSE is about double for move_3D as it is for move_1D data.

# In[137]:


sig2tree_3D = DecisionTreeRegressor(max_depth=3)
x = data[['signal2', 'liq_score', 'bondlength', 'quantity_bonds']]
y = data['move_3D']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.20)


# In[138]:


x


# In[139]:


sig2tree_3D.fit(x,y)


# In[140]:


y_pred = sig2tree_3D.predict(x)


# In[141]:


mse = mean_squared_error(y, y_pred)
rmse_tree3 = math.sqrt(mse)
rmse_tree3


# In[142]:


sig2tree_3D.fit(xtrain, ytrain)

ypred = sig2tree_3D.predict(xtest)


# In[209]:


mse = mean_squared_error(ytest, ypred)
rmse_tree4 = math.sqrt(mse)
rmse_tree4


# X[0]= signal2  X[1]= liq_score X[2]= bondlength X[3]= quantity_bonds

# In[144]:


tree.plot_tree(sig2tree_3D)
plt.rcParams["figure.figsize"]=15,15
plt.show()

