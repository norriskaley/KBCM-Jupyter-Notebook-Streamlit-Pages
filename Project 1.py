#!/usr/bin/env python
# coding: utf-8

# # Getting dataframe and it all set up

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns


# In[5]:


import os


# In[6]:


os.getcwd()


# In[7]:


os.chdir(r'C:\Users\NORRIKA\OneDrive - Key Corp\Documents\Project')


# In[8]:


pd.read_csv('BondSignalData.csv')


# In[9]:


data = pd.read_csv('BondSignalData.csv')


# In[7]:


data.dtypes


# # turning request time into date time

# In[10]:


data.request_time


# In[11]:


type(data['request_time'][0])


# In[12]:


data['request_time'][0]


# In[13]:


import dateutil.parser as dparser
dparser.parse('1/4/21 8:32 AM')
dparser.parse('1/4/21 8:32 AM').date()


# In[14]:


data.request_time.apply(dparser.parse().date())


# In[15]:


data['rqdatetime'] = pd.to_datetime(data.request_time)


# In[16]:


data.head()


# In[17]:


data.dtypes


# In[18]:


import datetime as dt


# In[19]:


data['rqdatetime'].dt.date


# In[20]:


data['rfqdate']=data['rqdatetime'].dt.date


# In[21]:


data.head()


# In[22]:


#plt.scatter(data['rfqdate'], data['signal1'])


# In[23]:


data.groupby(['rfqdate']).signal1.mean()


# In[24]:


data['signal1dailyavg'] = data.groupby(['rfqdate'])['signal1'].transform('mean')
data['signal2dailyavg'] = data.groupby(['rfqdate'])['signal2'].transform('mean')
data['signal3dailyavg'] = data.groupby(['rfqdate'])['signal3'].transform('mean')


# In[25]:


data.head()


# In[26]:


plt.scatter(data['rfqdate'], data['signal1dailyavg'])


# In[31]:


plt.scatter(data['month'], data['signal1dailyavg'])


# In[32]:


#days_sorted = sorted(data.rfqdate, key=lambda day: datetime.strptime(day, "%d-%b-%Y"))


# In[33]:


data.dtypes


# In[29]:


data['year']=pd.to_datetime(data.request_time).dt.year
data['month']=data.rqdatetime.dt.month
data['day']=pd.to_datetime(data.request_time).dt.day
data['hour']=pd.to_datetime(data.request_time).dt.hour
data['minute']=pd.to_datetime(data.request_time).dt.minute


# In[30]:


data['day']= data['day'].apply(lambda x: '{0:0>2}'.format(x)) #turning day to two digits so 1-04 comes before 1-11


# In[27]:


data.head()


# In[31]:


data.dtypes


# In[34]:


data['month_day']= (data['month'].astype(str) + "-" + data['day'].astype(str)) #putting month & day together


# In[33]:


data.head()


# # combining month 4 w 3 

# In[35]:


data['month'] = data['month'].replace(4, 3)


# In[31]:


data[(data['month'] == 3)] 


# # Box plots (month & hour vs. signal[1,2,3])

# In[36]:


sns.boxplot(x=data['month'], y=data['signal1'])


# In[37]:


sns.boxplot(x=data['month'], y=data['signal2'])


# In[38]:


sns.boxplot(x=data['month'], y=data['signal3'])


# In[39]:


sns.boxplot(x=data['hour'], y=data['signal1'])


# In[40]:


sns.boxplot(x=data['hour'], y=data['signal2'])


# In[41]:


sns.boxplot(x=data['hour'], y=data['signal3'])


# In[42]:


hour=[8,11,12,13,14,15,16,17,18,19,20,21,22]
for h in hour:
    print(h, data[data['hour']== h]['move_1D'].corr(data[data['hour']== h]['signal3']))


# # Trying to look @ industry sectors

# In[43]:


data.groupby('industrySector').mean()


# In[44]:


data.groupby('industrySector').count()


# In[45]:


data.groupby(['industrySector'])['move_1D'].describe()


# In[46]:


data.groupby(['industrySector'])['signal1'].describe()


# # come back to these. not done with 

# In[47]:


data[['industrySector','month_day','signal1','signal2','signal3','move_1D','move_3D']]


# In[48]:


deeperindustrysector=data[['industrySector','month_day','signal1','signal2','signal3','move_1D','move_3D']]


# In[49]:


deeperindustrysector.groupby(['industrySector']).max()


# In[50]:


deeperindustrysector.groupby(['industrySector']).describe()


# # looking at corr between sectors and move_1D and signals
# ###can use deeperindustrysector or data; same info in each dataframe

# ## signal3 sector corr

# In[53]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for sector in industrySector:
       print(sector, data[data['industrySector']== sector]['move_1D'].corr(data[data['industrySector']== sector]['signal3']))


# ## signal1 sector corr

# In[202]:


sns.scatterplot(data[data['industrySector']=='Consumer Non-cyclical']['move_1D'],data[data['industrySector']=='Consumer Non-cyclical']['signal1'] )


# In[54]:


industrySector=['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']
for sector in industrySector:
       print(sector, data[data['industrySector']== sector]['move_1D'].corr(data[data['industrySector']== sector]['signal1']))


# In[36]:


data[(data['industrySector'] == 'Consumer Non-cyclical')]


# In[37]:


data[(data['industrySector'] == 'Consumer Cyclical')]


# # month

# In[272]:


data[(data['month'] == 2)]


# In[127]:


data.groupby('month').count() #month 4 has significantly less than all the other months so combined month 4 with 3


# In[56]:


sns.lineplot(x=data['month'], y=data['signal1'])


# # joint plots

# In[54]:


data.groupby('hour').count()  #2-8pm is the bueist time to ask for an RFQ


# In[155]:


sns.jointplot(x=data['hour'], y= data['signal1'], kind ='kde') #7:00pm is the busiest rfq hour, most dense at hour 19


# In[159]:


sns.lineplot(x=data['move_1D'], y=data['signal1dailyavg'])


# In[55]:


sns.lineplot(x=data['move_1D'], y=data['signal3']) #showcasing relationship below


# In[37]:


fig, ax = plt.subplots()
sns.jointplot(x=data['signal3'], y= data['move_1D'], ax=ax)
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal3')
plt.show()


# In[174]:


fig, ax = plt.subplots()
sns.jointplot(x=data['move_1D'], y= data['signal3'], kind ='kde', ax=ax)
ax.set_xlim(-1,1)
ax.set_ylabel('signal3')
ax.set_xlabel('move_1D')
plt.show()


# In[180]:


fig, ax = plt.subplots()
sns.jointplot(x=data['move_1D'], y= data['signal2'], kind ='kde', ax=ax)
ax.set_xlim(-1,1)
ax.set_ylabel('signal2')
ax.set_xlabel('move_1D')
plt.show()


# In[62]:


sns.jointplot(x=data['move_1D'], y= data['signal2'])
ax.set_xlim(-1,1)
ax.set_ylabel('signal2')
ax.set_xlabel('move_1D')
plt.show() #another way of looking at graph above, scatter plot instead of heat/density map


# In[1]:


fig, ax = plt.subplots()
sns.jointplot(x=data['signal1'], y= data['move_1D'], kind ='kde', ax=ax)
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal1')
plt.show()


# In[61]:


sns.jointplot(x=data['signal1'], y= data['move_1D'])
ax.set_xlim(-1,1)
ax.set_ylabel('move_1D')
ax.set_xlabel('signal1')
plt.show()


# In[62]:


data.describe()


# In[63]:


#plt.scatter(data['request_time'], data['signal1'])


# In[150]:


#sns.distplot(data.signal1, kde=True, color="g")


# In[65]:


#sns.lineplot(x=data.request_time, y=data.signal1)


# # looking at when the move_1D is >1 or <1

# In[68]:


data[(data['move_1D'] > 0)]


# In[75]:


data[(data['move_1D'] < 0)]


# In[90]:


fig, ax = plt.subplots()
sns.jointplot(x= data['signal3'], y=(data['move_1D'] < 0), kind ='kde', ax=ax)
#DON'T KNOW IF THIS SHOWS ANYTHING as move_1D is supposd to be less than 0


# In[93]:


fig, ax = plt.subplots()
sns.jointplot(x=data['signal3'], y=(data['move_1D']), kind ='kde', ax=ax)


# In[92]:


fig, ax = plt.subplots()
sns.jointplot(x= (data['signal3']>0), y=(data['move_1D']), kind ='kde', ax=ax)


# In[271]:


data[(data['move_1D'] < 1) & (data['move_1D'] > -1)]


# In[76]:


posmove_1D=data[(data['move_1D'] > 0)]


# In[77]:


possignal1= data[(data['signal1'] > 0)]


# In[83]:


sns.lineplot(x=posmove_1D, y=possignal1) #i want to somehow see when signal one is poitive is move_1D pos too?


# In[194]:


data.groupby(['move_1D'])['signal1'].transform('mean')


# # correlations

# In[101]:


data.corr()


# In[102]:


data['move_1D'].corr(data['signal3'])


# ### looking for correlations between categorical variables

# In[136]:


# Cross tabulation between industrySector and ticker
CrosstabResult=pd.crosstab(index=data['industrySector'],columns=data['ticker'])
print(CrosstabResult)
 
# importing the required function
from scipy.stats import chi2_contingency
 
# Performing Chi-sq test
ChiSqResult = chi2_contingency(CrosstabResult)
 
# P-Value is the Probability of H0 being True
# If P-Value&gt;0.05 then only we Accept the assumption(H0)
 
print('The P-Value of the ChiSq Test is:', ChiSqResult[1])
#0-0.5 = variables related; greater than 0.05 = not correlated, did for dun as knew these would be correlated


# In[138]:


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


# # change maturity bond to number only

# In[60]:


data['bondlength'] = data.mat_bucket.str.strip("B")


# In[61]:


data['bondlength']=data.bondlength.astype(int)


# In[62]:


data[['bondlength','month_day','signal1','signal2','signal3','move_1D','move_3D']]


# ### looking at graphs with bondlength to find any correlations

# In[99]:


fig, ax = plt.subplots()
sns.jointplot(x=data['move_3D'], y= data['bondlength'], kind ='kde', ax=ax)
ax.set_xlim(-1,1)
ax.set_ylim(0,31)
ax.set_ylabel('bondlength')
ax.set_xlabel('move_3D')
plt.show() #did with signal1 and signal2 and these mean nothing, no relationship 


# In[183]:


data.bondlength.unique()


# In[195]:


data.groupby('bondlength').count()


# ## looking at each bondlength compared to signal3 

# In[340]:


bondlength= [2,3,5,10,30]
for bond in bondlength:
    print(bond, data[data['bondlength']== bond]['move_1D'].corr(data[data['bondlength']== bond]['signal3']))


# In[278]:


X = data[data['bondlength']== 30]['signal3']
Y = data[data['bondlength']== 30]['move_1D']

X = sm.add_constant(X)

model30 = sm.OLS(Y, X).fit()
predictions30 = model30.predict(X) 

print_model30 = model30.summary()
print(print_model30) #the variation of move_1D is explained 70.3% of the time by signal3


# In[280]:


np.sqrt(model30.resid_mse)


# In[299]:


sns.lineplot(x=data[data['bondlength']== 30]['signal3'], y=predictions30)
sns.lineplot(x=data[data['bondlength']== 30]['signal3'], y=-0.0170, color = 'red')
plt.scatter(data[data['bondlength']== 30]['signal3'], data[data['bondlength']== 30]['move_1D'], color = 'black')


# In[ ]:


#looking at LinReg of best bondlength=30 compared to worst=2 for signal3


# In[300]:


X = data[data['bondlength']== 2]['signal3']
Y = data[data['bondlength']== 2]['move_1D']

X = sm.add_constant(X)

modelB2 = sm.OLS(Y, X).fit()
predictionsB2 = modelB2.predict(X) 

print_modelB2 = modelB2.summary()
print(print_modelB2) #the variation of move_1D is explained 70.3% of the time by signal3


# In[302]:


sns.lineplot(x=data[data['bondlength']== 2]['signal3'], y=predictionsB2)
sns.lineplot(x=data[data['bondlength']== 2]['signal3'], y=.0315, color = 'red')
plt.scatter(data[data['bondlength']== 2]['signal3'], data[data['bondlength']== 2]['move_1D'], color = 'black')


# ## looking at each bondlength compared to signal1 

# In[341]:


bondlength= [2,3,5,10,30]
for bond in bondlength:
    print(bond, data[data['bondlength']== bond]['move_1D'].corr(data[data['bondlength']== bond]['signal1']))


# In[192]:


sns.boxplot(x=data['bondlength'], y=data['signal1'])


# In[193]:


sns.boxplot(x=data['bondlength'], y=data['signal2'])


# In[194]:


sns.boxplot(x=data['bondlength'], y=data['signal3'])


# see below with the following two graphs but i do not think bondlength has anything to do with how they move after 1 and 3 days. seem like the same graphs

# In[195]:


sns.boxplot(x=data['bondlength'], y=data['move_1D'])


# In[206]:


sns.boxplot(x=data['bondlength'], y=data['move_3D'])


# In[352]:


sns.lineplot(x=data['industrySector'], y=data['bondlength']) #does the bond maturity have anything to do with the sector??
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[58]:


sns.lineplot(x=data['industrySector'], y=data['quantity_bonds']) #does the bond maturity have anything to do with the sector??
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# # look at quantity of bonds

# In[347]:


data.quantity_bonds.sort_values().unique()


# In[230]:


#sns.boxplot(x=data['month_day'], y=data['quantity_bonds']) #does quantity have anything to do with time of time? signal?


# In[208]:


data.groupby('quantity_bonds').count()


# ### binning as the dist is right skewed

# In[255]:


sns.distplot(data['quantity_bonds'])


# In[61]:


bins = [-250, 250, 750, 1250,3000]
data['binned_quantity'] = pd.cut(data['quantity_bonds'], bins, labels = [1,2,3,4])
print (data)
data.head()


# In[47]:


sns.distplot(data[(data['binned_quantity'] == 1)]['move_1D'], kde = False, color ='red', bins = 30) 


# In[48]:


sns.distplot(data[(data['binned_quantity'] == 1)]['signal1'], kde = False, color ='red', bins = 30) 


# In[52]:


sns.distplot(data['signal1'], kde = False, color ='red', bins = 30) 


# In[258]:


data['move_1D'].corr(data['binned_quantity']==1)


# In[67]:


plt.scatter(data['signal1'],data['move_1D'])


# In[66]:


plt.scatter(data[(data['binned_quantity'] == 1)]['signal1'],data[(data['binned_quantity'] == 1)]['move_1D'])


# In[262]:


sns.distplot(data[(data['binned_quantity'] == 2)]['move_1D'], kde = False, color ='red', bins = 30) 


# In[53]:


sns.distplot(data[(data['binned_quantity'] == 2)]['signal1'], kde = False, color ='red', bins = 30) 


# In[65]:


plt.scatter(data[(data['binned_quantity'] == 2)]['signal1'],data[(data['binned_quantity'] == 2)]['move_1D'])


# In[350]:


binned_quantity= [1,2,3,4] #1=(-250,250) 2=(250,750) 3=(750,1250) 4=(1250,3000)
for q in binned_quantity:
    print(q, data[data['binned_quantity']== q]['move_1D'].corr(data[data['binned_quantity']== q]['signal3']))


# In[349]:


binned_quantity= [1,2,3,4] #1=(-250,250) 2=(250,750) 3=(750,1250) 4=(1250,3000)
for q in binned_quantity:
    print(q, data[data['binned_quantity']== q]['move_1D'].corr(data[data['binned_quantity']== q]['signal1']))


# In[166]:


plt.scatter(data[(data['binned_quantity'] == 4)]['signal1'],data[(data['binned_quantity'] == 4)]['move_1D'])


# # 	looking at ticker

# In[110]:


data.ticker.unique()


# In[111]:


data.groupby('ticker').describe()


# In[112]:


data.groupby(['ticker'])['move_1D'].describe() #looking at the move_1D data for each individual ticker


# In[237]:


data[(data['ticker'] == 'XEL')] 


# ### grouping tickers with their sectors and looking at each signal

# In[240]:


data.groupby(['ticker'])['industrySector'].describe() #what sector each ticker is


# In[271]:


tickersig1df= data.groupby(['ticker','industrySector'])['signal1'].describe() 


# In[272]:


tickersig1df


# In[273]:


#mu = mean #std = standard deviation
mu, std = norm.fit(tickersig1df['mean'])
plt.hist(tickersig1df['mean'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() 


# In[214]:


#signal2


# In[274]:


tickersig2df= data.groupby(['ticker','industrySector'])['signal2'].describe() 


# In[275]:


tickersig2df


# In[276]:


#mu = mean #std = standard deviation
mu, std = norm.fit(tickersig2df['mean'])
plt.hist(tickersig2df['mean'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() 


# In[215]:


#signal3


# In[33]:


tickersig3df= data.groupby(['ticker','industrySector'])['signal3'].describe() 


# In[34]:


tickersig3df


# # COME BACK TO LOOK AT THIS BELOW!!!

# In[279]:


#mu = mean #std = standard deviation
mu, std = norm.fit(tickersig3df['mean'])
plt.hist(tickersig3df['mean'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() 


# In[216]:


#move_1D


# In[259]:


ticker1Ddf=data.groupby(['ticker','industrySector'])['move_1D'].describe() 


# In[260]:


ticker1Ddf


# In[171]:


data.groupby(['ticker','industrySector'])['move_1D']


# In[262]:


#mu = mean #std = standard deviation
mu, std = norm.fit(ticker1Ddf['count'])
plt.hist(ticker1Ddf['count'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() #count should be the same for move_3D as number of each ticker/industry sector isn't changing


# In[264]:


#mu = mean #std = standard deviation
mu, std = norm.fit(ticker1Ddf['mean'])
plt.hist(ticker1Ddf['mean'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() #looking at the mean move_1D of each ticker/industry sector?!


# In[263]:


ticker1Ddf.groupby('industrySector').describe() #showing how many tickers are of each sector


# In[265]:


ticker3Ddf=data.groupby(['ticker','industrySector'])['move_3D'].describe() 


# In[266]:


ticker3Ddf


# In[267]:


#mu = mean #std = standard deviation
mu, std = norm.fit(ticker3Ddf['mean'])
plt.hist(ticker3Ddf['mean'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show()


# # trying to see if ticker influences corr between move_1D and signals

# In[275]:


data.ticker.unique()


# In[318]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for t in ticker:
    print(t, data[data['ticker']== t]['move_1D'].corr(data[data['ticker']== t]['signal3']))


# In[53]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
d = {}
for t in ticker:
    d[t]=data[data['ticker']== t]['move_1D'].corr(data[data['ticker']== t]['signal3'])


# In[54]:


d


# ### signal3 is definitely better at predicting move_1D data for some tickers better than others. see values above

# In[319]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for t in ticker:
    print(t, data[data['ticker']== t]['move_1D'].corr(data[data['ticker']== t]['signal2']))


# In[320]:


ticker=['MSFT', 'GP', 'KSS', 'DOW', 'JPM', 'COST', 'RY', 'CVS', 'TFC',
       'RSG', 'BA', 'PNC', 'AMGN', 'PCG', 'GM', 'GLENLN', 'MRK', 'PYPL',
       'MS', 'PKI', 'VLO', 'BK', 'LMT', 'CAG', 'VMW', 'HUM', 'NEE',
       'KEYS', 'DE', 'CSX', 'ROP', 'MKL', 'UDR', 'PFE', 'WMB', 'KDP',
       'DXC', 'VST', 'TSN', 'NSC', 'GE', 'DUK', 'BAYNGR', 'SRC', 'KMI',
       'O', 'FBHS', 'QCOM', 'AIG', 'XEL', 'CAT', 'MCO', 'BNSF']
for t in ticker:
    print(t, data[data['ticker']== t]['move_1D'].corr(data[data['ticker']== t]['signal1']))


# # looking at liq_score

# In[120]:


data['liq_score']


# In[134]:


data['liq_score'].describe()


# In[62]:


from scipy.stats import norm


# In[63]:


mu, std = norm.fit(data['liq_score'])


# In[149]:


plt.hist(data['liq_score'], bins=25, density=True, alpha=0.6, color='b')


# In[181]:


sns.distplot(data['liq_score'], color ='red', bins = 25)
plt.xlim(-500, 3500, 250) 


# In[184]:


sns.distplot(data[(data['liq_score'] < 500)]['move_1D'], kde = False, color ='red', bins = 30)


# In[148]:


sns.distplot(data[(data['liq_score'] >500)]['move_1D'], kde = False, color ='red', bins = 30) 


# In[160]:


data[data['liq_score'] < 500] #trying to see how many have a liq_score < 500 as graph is right skewed


# In[165]:


#mu = mean #std = standard deviation
mu, std = norm.fit(data['liq_score'])
plt.hist(data['liq_score'], bins=10, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show() #first fit value is mean, second is std


# ### binning liq_score 

# In[67]:


bins = [0, 605, 1500, 2000, 3000]
data['binnedliq_score'] = pd.cut(data['liq_score'], bins, labels = [1,2,3,4])
print (data)
data.head()


# In[68]:


binnedliq_score=[1,2,3,4] #1=(0, 605) 2= (605,1500) 3=(1500,2000) 4=(2000,3000)
for b in binnedliq_score:
    print(b, data[data['binnedliq_score']== b]['move_1D'].corr(data[data['binnedliq_score']== b]['signal3']))


# In[69]:


binnedliq_score=[1,2,3,4] #1=(0, 605) 2= (605,1500) 3=(1500,2000) 4=(2000,3000)
for b in binnedliq_score:
    print(b, data[data['binnedliq_score']== b]['move_1D'].corr(data[data['binnedliq_score']== b]['signal1']))


# In[70]:


#i do not think liq_score has anything to do with how well signal perfroms in relation to move_1D info
#HOWEVER the corr is higher for bin 1 than bin2 for signal1


# In[118]:


sns.distplot(data[(data['binnedliq_score'] == 1)]['signal1'], kde = False, color ='red', bins = 30) 


# In[119]:


sns.distplot(data[(data['binnedliq_score'] == 1)]['move_1D'], kde = False, color ='red', bins = 30) 


# In[121]:


plt.scatter(data[(data['binnedliq_score'] == 1)]['signal1'],data[(data['binnedliq_score'] == 1)]['move_1D'])


# In[134]:


plt.scatter(data[(data['binnedliq_score'] == 1)]['signal3'],data[(data['binnedliq_score'] == 1)]['move_1D'])


# In[135]:


plt.scatter(data[(data['binnedliq_score'] == 4)]['signal3'],data[(data['binnedliq_score'] == 4)]['move_1D'])
#i do not think liq_score has anything to do with how well signal perfroms in relation to move_1D info(at least for signal3)


# In[127]:


sns.distplot(data[(data['binnedliq_score'] == 2)]['signal1'], kde = False, color ='red', bins = 30) #start of bin 2 data


# In[126]:


sns.distplot(data[(data['binnedliq_score'] == 2)]['signal3'], kde = False, color ='red', bins = 30) 


# In[128]:


sns.distplot(data[(data['binnedliq_score'] == 2)]['move_1D'], kde = False, color ='red', bins = 30) 


# In[129]:


plt.scatter(data[(data['binnedliq_score'] == 2)]['signal1'],data[(data['binnedliq_score'] == 2)]['move_1D'])


# In[136]:


sns.distplot(data[(data['binnedliq_score'] == 3)]['signal1'], kde = False, color ='red', bins = 30) #start of bin 2 data


# In[137]:


sns.distplot(data[(data['binnedliq_score'] == 3)]['signal2'], kde = False, color ='red', bins = 30) #start of bin 2 data


# In[138]:


sns.distplot(data[(data['binnedliq_score'] == 3)]['signal3'], kde = False, color ='red', bins = 30) #start of bin 2 data


# In[139]:


sns.distplot(data[(data['binnedliq_score'] == 3)]['move_1D'], kde = False, color ='red', bins = 30) #start of bin 2 data


# In[141]:


plt.scatter(data[(data['binnedliq_score'] == 3)]['signal1'],data[(data['binnedliq_score'] == 3)]['move_1D'])


# In[157]:


plt.scatter(data[(data['binnedliq_score'] == 4)]['signal1'],data[(data['binnedliq_score'] == 4)]['move_1D'])


# # daily average calc 

# In[71]:


data['move1Ddailyavg'] = data.groupby(['rfqdate'])['move_1D'].transform('mean')
data['move3Ddailyavg'] = data.groupby(['rfqdate'])['move_3D'].transform('mean')


# In[72]:


data.head()


# In[73]:


dailyaverages = data[['month_day','signal1dailyavg','signal2dailyavg','signal3dailyavg','move1Ddailyavg','move3Ddailyavg']]


# In[74]:


max_dailyaverages=dailyaverages.groupby(['month_day']).max()


# In[75]:


max_dailyaverages.index


# In[76]:


month_day= ['1-04', '1-05', '1-06', '1-07', '1-08', '1-11', '1-12', '1-13', '1-14',
       '1-15', '1-18', '1-19', '1-20', '1-21', '1-22', '1-25', '1-26', '1-27',
       '1-28', '1-29', '2-01', '2-02', '2-03', '2-04', '2-05', '2-08', '2-09',
       '2-10', '2-11', '2-12', '2-16', '2-17', '2-18', '2-19', '2-23', '2-24',
       '2-25', '2-26', '3-01', '3-02', '3-03', '3-04', '3-05', '3-08', '3-09',
       '3-10', '3-11', '3-12', '3-15', '3-16', '3-17', '3-18', '3-19', '3-22',
       '3-23', '3-24', '3-25', '3-26', '3-29', '3-30', '3-31', '4-01']
for d in month_day:
    print(d, data[data['month_day']== d]['move_1D'].corr(data[data['month_day']== d]['signal3']))


# In[58]:


dailyaverages.groupby(['month_day']).head(1)


# In[59]:


data['move3Ddailyavg'].corr(data['signal2dailyavg'])


# In[63]:


mu, std = norm.fit(data['signal1dailyavg'])
plt.hist(data['signal1dailyavg'], bins=25, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show()


# # box plots w avgs

# In[65]:


sns.boxplot(x=data['month'], y=data['move1Ddailyavg'])


# In[66]:


sns.boxplot(x=data['month'], y=data['signal1dailyavg'])


# In[67]:


sns.boxplot(x=data['month'], y=data['signal2dailyavg'])


# In[68]:


sns.boxplot(x=data['month'], y=data['signal3dailyavg'])


# # explore

# ### looking at when all move and signals >= 0

# In[296]:


data.loc[(data["move_3D"]>=0) & (data["move_1D"]>=0) & (data["signal1"]>=0) & (data["signal2"]>=0) & (data["signal3"]>=0), ["move_1D","signal1", "signal2", "signal3","move_3D", "month_day", "quantity_bonds", "liq_score"]]


# In[306]:


data_sorted = data.sort_values(['move_1D','signal3'], ascending=False)
data_sorted[['move_1D','signal3']].head()


# In[ ]:


#%matplotlib inline
#data.hist(column="month_day",by="move_1D",bins=30) didnt work


# In[ ]:


sns.distplot(data.move_1D, ax=ax_hist)


# In[308]:


sns.distplot(data['move1Ddailyavg'], kde = False, color ='red', bins = 30)


# In[309]:


sns.distplot(data['move3Ddailyavg'], kde = False, color ='red', bins = 30)


# In[310]:


sns.distplot(data['signal1dailyavg'], kde = False, color ='red', bins = 30)


# In[313]:


sns.distplot(data['signal2dailyavg'], kde = False, color ='red', bins = 30)


# In[317]:


sns.distplot(data['signal3dailyavg'], kde = False, color ='red', bins = 30)


# In[350]:


sns.distplot(data['industrySector'], kde = False, color ='red', bins = 30)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[326]:


sns.distplot(data['liq_score'], kde = False, color ='red', bins = 30)


# In[157]:


mu, std = norm.fit(data['signal2dailyavg'])
plt.hist(data['signal2dailyavg'], bins=25, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)
  
plt.show()


# In[180]:


data.head()


# # dist plots for move

# In[299]:


sns.distplot(data['move_1D']) #can filter data frame to get when sector is energy example AARON"S HELP!!!! see two graphs below this one


# In[291]:


sns.distplot(data['move_3D'])


# # dist plots for signals

# In[322]:


sns.distplot(data['signal1'])


# In[323]:


sns.distplot(data['signal2'])


# In[324]:


sns.distplot(data['signal3'])


# ### splitting  signal3 into 3 different bins to look at those humps seperately. is there a better correlation between the signal and that specific highly correlated bin?

# In[77]:


bins = [-1.25, -0.75, 0.75, 1.25]
data['binnedsig3'] = pd.cut(data['signal3'], bins, labels = [1,2,3])
print(data)
data.head()


# In[264]:


data[(data['binnedsig3'] == 1)]


# In[241]:


sns.distplot(data[(data['binnedsig3'] == 1)]['move_1D'], kde = False, color ='red', bins = 30) #(-1.25, -0.75]


# In[242]:


data['move_1D'].corr(data['binnedsig3']==1)


# In[243]:


sns.distplot(data[(data['binnedsig3'] == 2)]['move_1D'], kde = False, color ='red', bins = 30) #(-0.75, 0.75]


# In[244]:


data['move_1D'].corr(data['binnedsig3']==2)


# In[248]:


sns.distplot(data[(data['binnedsig3'] == 3)]['move_1D'], kde = False, color ='red', bins = 30) #(0.75, 1.25]


# In[245]:


data['move_1D'].corr(data['binnedsig3']==3)


# ## ^What does this mean/do??? **found that in those humps, the corr between signal3 and move_1D is greater!

# # look at dist plot for each sector

# In[312]:


data['industrySector'].unique()


# In[316]:


sns.distplot(data[data['industrySector'] == "Consumer Cyclical"]['move_1D']) #data set is now filtered by sector


# In[313]:


sns.distplot(data[data['industrySector'] == "Technology"]['move_1D']) #data set is now filtered by sector


# In[314]:


sns.distplot(data[data['industrySector'] == "Basic Materials"]['move_1D'])


# In[317]:


sns.distplot(data[data['industrySector'] == "Financial"]['move_1D'])


# In[318]:


sns.distplot(data[data['industrySector'] == "Consumer Non-cyclical"]['move_1D'])


# In[319]:


sns.distplot(data[data['industrySector'] == "Industrial"]['move_1D'])


# In[320]:


sns.distplot(data[data['industrySector'] == "Utilities"]['move_1D'])


# In[321]:


sns.distplot(data[data['industrySector'] == "Energy"]['move_1D']) 


# In[332]:


data.head()


# In[315]:


#data[data["industrySector"] == "Financial"]["move_1D"].plot(kind="hist") #Histograms by sector


# # look at dist plot for each bondlength

# In[340]:


data['bondlength'].unique()


# In[70]:


x1 = list(data[data['bondlength'] == 2]['move_1D'])
x2 = list(data[data['bondlength'] == 3]['move_1D'])
x3 = list(data[data['bondlength'] == 5]['move_1D'])
x4 = list(data[data['bondlength'] == 10]['move_1D'])
x5 = list(data[data['bondlength'] == 30]['move_1D'])

colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
bondlengths = [2,3,5,10,30]
plt.hist([x1, x2, x3, x4, x5], bins = int(12/1), normed=True,
         color = colors, label=bondlengths)

plt.legend()
plt.xlabel('move_1D')
plt.ylabel('Normalized __')
plt.title('Side-by-Side Histogram with Multiple bond maturities')
# For the plot calls, we specify the binwidth by the number of bins. For this plot, I will use bins that are 5 minutes
#in length, which means that the number ofbins will be the range of the data (from -60 to 120 minutes) divided by the 
#binwidth, 5 minutes ( bins = int(180/5)).


# In[336]:


sns.distplot(data[data['bondlength'] == 2]['move_1D'])


# In[335]:


sns.distplot(data[data['bondlength'] == 3]['move_1D'])


# In[337]:


sns.distplot(data[data['bondlength'] == 5]['move_1D'])


# In[338]:


sns.distplot(data[data['bondlength'] == 10]['move_1D'])


# In[339]:


sns.distplot(data[data['bondlength'] == 30]['move_1D'])


# In[ ]:


##!can repeat each of these bondlengt or insustry sectors for move_1D and move_3D but just focus on 1D and can always repeat the process with 3D later per Aaronh


# In[341]:


sns.distplot(data[data['bondlength'] == 30]['move_3D'])


# # binning/buckets

# In[78]:


bins = [-1, -0.5, 0, 0.5, 1]
data['binnedsig1'] = pd.cut(data['signal1'], bins, labels = [1,2,3,4])
print (data)
data.head()


# In[199]:


data['binnedsig1'].dtypes


# In[101]:


data[(data['binnedsig1'] == 1)] 


# In[200]:


sns.distplot(data[(data['binnedsig1'] == 1)]['move_1D'], kde = False, color ='red', bins = 30) #(-1.0, -0.5]


# In[106]:


sns.distplot(data[(data['binnedsig1'] == 2)]['move_1D'], kde = False, color ='red', bins = 30) #(-0.5, 0.0]


# In[134]:


sns.distplot(data[(data['binnedsig1'] == 3)]['move_1D'], kde = False, color ='red', bins = 30) #(0.0, 0.5]


# In[135]:


sns.distplot(data[(data['binnedsig1'] == 4)]['move_1D'], kde = False, color ='red', bins = 30) #(0.5, 1.0]


# In[97]:


plt.scatter(data['signal1'], data['move_1D'])


# In[137]:


#fig,ax=plt.subplots()
#sns.lineplot(x=data['binnedsig1'], y=data['move_1D'])
## sns.lineplot(x=data['signal1'], y=data['signal2'])
#plt.show()


# ### looking at what sector is the highest in that signal category

# In[222]:


data['industrySector'].unique()


# In[345]:


sns.distplot(data[(data['binnedsig1'] == 1)]['industrySector'], kde = False, color ='red', bins = 30) #(-1.0, -0.5]
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45) #consumer non-cyclical = highest


# In[346]:


sns.distplot(data[(data['binnedsig1'] == 2)]['industrySector'], kde = False, color ='red', bins = 30)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45) #consumer non-cyclical = highest


# In[348]:


sns.distplot(data[(data['binnedsig1'] == 3)]['industrySector'], kde = False, color ='red', bins = 30)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45) #consumer non-cyclical = highest


# In[349]:


sns.distplot(data[(data['binnedsig1'] == 4)]['industrySector'], kde = False, color ='red', bins = 30)
plt.tick_params(axis='x', which='major', labelsize=10, rotation=45)


# In[286]:


data['industrySector'].unique()


# # correlation matrix/heat map

# In[287]:


# Finding and plotting the correlation for
# the independent variables

# adjust plot
sns.set(rc={'figure.figsize': (14, 5)})

# assign data
ind_var = ['Technology', 'Basic Materials', 'Consumer Cyclical', 'Financial',
       'Consumer Non-cyclical', 'Industrial', 'Utilities', 'Energy']

# illustrate heat map.
sns.heatmap(data.select_dtypes(include='number').corr(), cmap=sns.cubehelix_palette(20, light=0.95, dark=0.15))


# # Machine Modeling

# In[226]:


data.head()


# In[236]:


sns.pairplot(data)


# In[108]:


data.columns.values.tolist()


# In[63]:


#cfm = columns for modeling
cfm = data[['cusip','industrySector','ticker','bondlength', 'quantity_bonds','liq_score','move_1D','move_3D','signal1', 'signal2', 'signal3','rqdatetime','month']]


# In[64]:


cfm.head()


# In[81]:


cfm_withavgs = data[['cusip','industrySector','ticker','bondlength', 'quantity_bonds','liq_score','move_1D','move_3D','signal1', 'signal1dailyavg', 'signal2', 'signal2dailyavg','signal3','signal3dailyavg','rqdatetime','month']]


# In[73]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']= 5, 4
sns.set_style('whitegrid')


# In[233]:


sns.pairplot(cfm)


# In[234]:


cfm.corr()


# In[294]:


cfm['signal3']


# ## wanting to look at linear regression of signal3 & move_1D when signal3 = -1 or 1
# 

# In[ ]:


box plot


# In[94]:


sns.boxplot(x=sig3atposandneg['signal3'], y=sig3atposandneg['move_1D'])


# In[95]:


sns.boxplot(x=sig3negone['signal3'], y=sig3negone['move_1D'])


# In[96]:


sns.boxplot(x=sig3posone['signal3'], y=sig3posone['move_1D'])


# In[69]:


cfm[(cfm['signal3'] == -1) | (cfm['signal3'] == 1)].sort_values('signal3')


# In[90]:


sig3atposandneg = cfm[(cfm['signal3'] == -1) | (cfm['signal3'] == 1)]


# In[71]:


sig3atposandneg


# In[74]:


x = sig3atposandneg['signal3']
y = sig3atposandneg['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model) #the variation of move_1D is explained 70.3% of the time by signal3


# In[75]:


sns.lineplot(x=sig3atposandneg['signal3'], y=predictions)
sns.lineplot(x=sig3atposandneg['move_1D'], y=0.0348, color = 'red')
plt.scatter(sig3atposandneg['signal3'], sig3atposandneg['move_1D'], color = 'black')


# In[85]:


sig3negone = cfm[(cfm['signal3'] == -1)]
sig3negone 


# In[84]:


sig3posone= cfm[(cfm['signal3'] == 1)]


# In[86]:


x = sig3negone['signal3']
y = sig3negone['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model) #the variation of move_1D is explained 70.3% of the time by signal3


# In[87]:


sns.lineplot(x=sig3negone['signal3'], y=predictions)
sns.lineplot(x=sig3negone['signal3'], y=0, color = 'red')
plt.scatter(sig3negone['signal3'], y = sig3negone['move_1D'], color = 'black')


# In[88]:


x = sig3posone['signal3']
y = sig3posone['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model) #the variation of move_1D is explained 70.3% of the time by signal3


# ## wanting to look at linear regression of signal3 & move_1D but not when signal3 = -1 or 1

# In[302]:


type(cfm)


# In[77]:


cfm[(cfm['signal3'] != -1) & (cfm['signal3'] != 1)].sort_values('signal3') #deleted rows when signal3 = -1 or 1


# In[78]:


sig3betweenposandneg1 = cfm[(cfm['signal3'] != -1) & (cfm['signal3'] != 1)]


# In[79]:


sig3betweenposandneg1 


# In[88]:


#first LinReg below but the one below this one is better!


# In[289]:


x = sig3betweenposandneg1['signal3']
y = sig3betweenposandneg1['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model) #the variation of move_1D is explained 70.3% of the time by signal3


# In[340]:


sns.lineplot(x=sig3betweenposandneg1['signal3'], y=predictions)
sns.lineplot(x=sig3betweenposandneg1['move_1D'], y=-0.0090, color = 'red')
plt.scatter(sig3betweenposandneg1['signal3'], sig3betweenposandneg1['move_1D'], color = 'black')


# ### linear regression of signal3 and move_1D 

# In[309]:


X = cfm['signal3']
Y = cfm['move_1D']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model) #the variation of move_1D is explained 70.3% of the time by signal3


# In[ ]:


#formula from above LinReg: move_1D = 1.2333 * signal3 +0.0017


# In[280]:


sns.lineplot(x=data['signal3'], y=predictions)
sns.lineplot(x=data['signal3'], y=.0017, color = 'red')
plt.scatter(data['signal3'], data['move_1D'], color = 'black')


# In[277]:


plt.scatter(data['signal3'], data['move_1D'])


# In[278]:


predictions #signal 3 vs predict and signal3 vs actual move_1D


# In[231]:


#Adj.R-squared = reflects the fit of the model. R-squared values range from 0 to 1, where a higher value
#generally indicates a better fit, assuming certain conditions are met.


# In[232]:


#const coefficient is your Y-intercept.
#signal3 coeff prepresents the change in the ouput Y due to a change of one unit in the signal3


# In[233]:


#std err reflects the level of accuracy of the coefficients. The lower it is, the higher is the level of accuracy
#P >|t| is your p-value. A p-value of less than 0.05 is considered to be statistically significant
#Confidence Interval represents the range in which our coefficients are likely to fall (with a likelihood of 95%)


# In[234]:


cfm_withavgs.corr()


# In[237]:


X = cfm[['signal1', 'signal2', 'signal3']]
Y = cfm['move_1D']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)


# In[ ]:


#signal1 coeff prepresents the change in the ouput Y due to a change of one unit in the signal1 
#signal2 coeff prepresents the change in the ouput Y due to a change of one unit in the signal2
#signal3 coeff prepresents the change in the ouput Y due to a change of one unit in the signal3
#output Y = move_1D


# ## regression (linear, poly, cublic spline)

# In[155]:


#linear regression


# In[110]:


A = cfm[['signal3']]
b = cfm[['move_1D']]


# In[111]:


from sklearn.model_selection import train_test_split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state = 1)


# In[112]:


sns.set(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
 
plt.figure(figsize=(10,8))
sns.scatterplot(x=A_train['signal3'], y=b_train['move_1D'], color="red",alpha=0.2)
plt.title("signal3 vs move_1D Training Dataset")
 
plt.figure(figsize=(10,8))
sns.scatterplot(x=A_test['signal3'], y=b_test['move_1D'], color="green",alpha=0.4)
plt.title("signal3 vs move_1D Testing Dataset")
 
plt.show()


# In[113]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(A_train,b_train)
 
print("Slope of the Regression Line is : ", lm.coef_)
print("Intercept of Regression Line is : ",lm.intercept_)
 
from sklearn.metrics import mean_squared_error
pred_test = lm.predict(A_test)
rmse_test = mean_squared_error(b_test, pred_test)
from math import sqrt
rmse_test = sqrt(mean_squared_error(b_test, pred_test))
 
print("Accuracy of Linear Regression on testing data is : ", rmse_test)


# In[114]:


plt.figure(figsize=(10,8))
sns.regplot(x=A_test['signal3'], y=b_test['move_1D'], ci=None, line_kws={"color": "red"})
plt.title("Regression Line for Testing Dataset")
plt.show()


# In[115]:


#polynomial regression


# In[116]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
 
A_train_poly = poly.fit_transform(A_train)
A_test_poly = poly.fit_transform(A_test)
pm = LinearRegression()
pm.fit(A_train_poly,b_train)
 
pred_test = pm.predict(A_test_poly)
rmse_test = mean_squared_error(b_test, pred_test)
from math import sqrt
rmse_test = sqrt(mean_squared_error(b_test, pred_test))
 
print("Accuracy of Polynomial Regression on testing data is : ",rmse_test)


# In[117]:


plt.figure(figsize=(10,8))
sns.regplot(x=A_test['signal3'], y=b_test['move_1D'], ci=None, line_kws={"color": "red"},order=2)
plt.title("Polynomial Regression Line for Testing Dataset")
plt.show()


# In[118]:


#cubic spline


# In[119]:


from patsy import dmatrix
transformed_x = dmatrix("bs(train, knots=(-1,0,1), degree=3, include_intercept=False)",
                        {"train": A_train},return_type='dataframe')
import statsmodels.api as sm
cs = sm.GLM(b_train, transformed_x).fit()
pred_test = cs.predict(dmatrix("bs(test, knots=(-1,0,1), include_intercept=False)",
                               {"test": A_test}, return_type='dataframe'))
rmse_test =mean_squared_error(b_test, pred_test)
from math import sqrt
rmse_test = sqrt(mean_squared_error(b_test, pred_test))
print("Accuracy for Cubic Spline on testing data is : ",rmse_test)
 
import numpy as np
plt.figure(figsize=(10,8))
xp = np.linspace(A_test.min(),A_test.max(), 100)
pred = cs.predict(dmatrix("bs(xp, knots=(-1,0,1), include_intercept=False)", 
                          {"xp": xp}, return_type='dataframe'))
sns.scatterplot(x=A_train['signal3'], y=b_train['move_1D'])
plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)', color='red')
plt.legend()
plt.title("Cubic Spline Regression Line for Testing Dataset")
plt.show()


# In[120]:


#lasso regression--not sure how to do this


# In[222]:


# from sklearn.linear_model import Lasso, LassoCV


# x, y = (cfm['signal3'], cfm['liq_score'], cfm['bondlength'], cfm['move_1D']

# model = Lasso.(fit(x, y))
# print(model)
# score = model.score(x, y)
# ypred = model.predict(xtest)
# mse = mean_squared_error(ytest,ypred)
# print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
#     .format(model.alpha, score, mse, np.sqrt(mse)))

# x_ax = range(len(ypred))
# plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show()

# alphas = [-1,-0.50,0, 0.5, 1]
# lassocv = LassoCV(alphas=alphas, cv=5).fit(x,y)
# print(lassocv)
# score = lassocv.score(x,y)
# ypred = lassocv.predict(xtest)
# mse = mean_squared_error(ytest,ypred)
# print("Alpha:{0:.2f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
#     .format(lassocv.alpha_, score, mse, np.sqrt(mse)))

# x_ax = range(len(xtest))
# plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show() 

