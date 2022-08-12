#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm


# In[3]:


os.chdir(r'C:\Users\NORRIKA\OneDrive - Key Corp\Documents\Project')


# In[4]:


data = pd.read_csv(r'C:\Users\NORRIKA\OneDrive - Key Corp\Documents\Project\BondSignalData.csv')


# In[5]:


data['rqdatetime'] = pd.to_datetime(data.request_time)
data['rqdatetime'].dt.date
data['rfqdate']=data['rqdatetime'].dt.date
data['year']=pd.to_datetime(data.request_time).dt.year
data['month']=data.rqdatetime.dt.month
data['day']=pd.to_datetime(data.request_time).dt.day
data['hour']=pd.to_datetime(data.request_time).dt.hour
data['minute']=pd.to_datetime(data.request_time).dt.minute
data['day']= data['day'].apply(lambda x: '{0:0>2}'.format(x)) #turning day to two digits so 1-04 comes before 1-11
data['month_day']= (data['month'].astype(str) + "-" + data['day'].astype(str)) #putting month & day together
data['month'] = data['month'].replace(4, 3)
data['bondlength'] = data.mat_bucket.str.strip("B")
data['bondlength']=data.bondlength.astype(int)


# In[6]:


bins = [-1.25, -0.75, 0.75, 1.25]
data['binnedsig3'] = pd.cut(data['signal3'], bins, labels = [1,2,3])
#print(data)
data.head()


# In[7]:


data['binnedsig3_binary'] = data['binnedsig3'].apply(lambda x: 1 if x == 2 else 0)
data['binnedsig3_binary']


# In[8]:


data['fin_binary'] = data['industrySector'].apply(lambda x: 1 if x == 'Financial' else 0)
data['fin_binary']


# # Trees

# In[9]:


clf = DecisionTreeRegressor(max_depth=3)
x = data[['quantity_bonds', 'liq_score','fin_binary', 'binnedsig3_binary', 'bondlength']]
y = data['move_1D']


# In[ ]:


x


# In[15]:


clf.fit(x,y)


# In[16]:


y_pred = clf.predict(x)


# In[17]:


mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)
rmse


# In[18]:


tree.plot_tree(clf)
plt.rcParams["figure.figsize"]=15,15
plt.show()


# In[36]:


data['tree_prediction'] = clf.predict(data[['quantity_bonds', 'liq_score','fin_binary', 'binnedsig3_binary', 'bondlength']])


# In[43]:


data.groupby(['tree_prediction']).count()


# ## is this tree important ? --playing around with a second tree & variables

# In[31]:


clf2 = DecisionTreeRegressor(max_depth=3)
x = data[['binnedliq_score', 'bondlength','signal3', 'sig3equal1', 'sig3equalneg1']]
y = data['move_1D']


# In[32]:


clf2.fit(x,y)


# In[33]:


y_pred = clf2.predict(x)


# In[34]:


tree.plot_tree(clf2)
plt.rcParams["figure.figsize"]=15,15
plt.show()


# In[38]:


data['tree_prediction2'] = clf2.predict(data[['binnedliq_score', 'bondlength','signal3', 'sig3equal1', 'sig3equalneg1']])


# In[41]:


data.groupby('tree_prediction2').count()


# In[ ]:





# In[ ]:


train test split


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)


# In[43]:


def k_Fold_Split(data, n_splits, shuffle, random_state, verbose=False):
    # Creating k-fold splitting 
    kfold = KFold(n_splits, shuffle, random_state)
    sizes = [0 , 0]
    for train, test in kfold.split(data):
        if verbose:
            print('train: indexes %s, val: indexes %s, size (training) %d, (val) %d' % (train, test, train.shape[0], test.shape[0]))
        sizes[0] += train.shape[0]
        sizes[1] += test.shape[0]
        
    return int(sizes[0] /n_splits), int(sizes[1]/n_splits)


# In[44]:


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
cv_strategy = LeaveOneOut()
# cross_val_score will evaluate the model 
scores = cross_val_score(clf, x, y, scoring='accuracy', cv=cv_strategy, n_jobs=-1)


# # BUCKETS

# ### BINNING LIQUIDITY SCORE BASED ON QCUT

# In[27]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8)


# In[28]:


data['binnedliq_score']


# In[29]:


data['binnedliq_score']= pd.qcut(data['liq_score'], q=8, labels=[1,2,3,4,5,6,7,8])


# In[30]:


data.groupby(['binnedliq_score']).count()


# ### BINNING MAT BUCKET BASED ON QCUT 

# In[24]:


data['binnedmat_bucket']= pd.qcut(data['bondlength'], q=4,  labels=['2-3 years', '5 years', '10 years', '30 years'])


# In[25]:


data.groupby(['binnedmat_bucket']).count()


# ### BINNING BOND QUANTITY BASED ON QCUT

# In[55]:


data['binnedquantity_bonds']= pd.qcut(data['quantity_bonds'], q=5, labels=['1-12', '12-50', '50-210', '210-600', '600-3000'])


# In[56]:


data.groupby(['binnedquantity_bonds']).count()


# # obtained best possible model

# In[99]:


data['sig3equal1'] = data['signal3'].apply(lambda x: 1 if x >= 1 else 0)
data['sig3equalneg1'] = data['signal3'].apply(lambda x: 1 if x <= -1 else 0)


# In[25]:


x = data[['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[26]:


model.params[0] #=constant from regression so can use in plotting later


# In[27]:


sns.lineplot(x = data['signal3'], y=predictions)
sns.lineplot(x = data['signal3'], y=model.params[0], color = 'red')
plt.scatter(data['signal3'],data['move_1D'], color = 'black')


# # now using model with buckets

# In[76]:


#most liquid, most traded, r-squared means what?? about 83.5% of the variation in move_1D can be explained by signal3


# ### Liquidity Score

# In[45]:


x = data[data['binnedliq_score']==1][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedliq_score']==1]['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[488]:


sns.lineplot(x = data[data['binnedliq_score']==1]['signal3'], y= predictions)
sns.lineplot(x =  data[data['binnedliq_score']==1]['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedliq_score']==1]['signal3'],data[data['binnedliq_score']==1]['move_1D'], color = 'black')


# In[ ]:


#middle of the road liquidity score


# In[489]:


x = data[data['binnedliq_score']==4][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedliq_score']==4]['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model) 


# In[490]:


sns.lineplot(x = data[data['binnedliq_score']==4]['signal3'], y= predictions)
sns.lineplot(x =  data[data['binnedliq_score']==4]['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedliq_score']==4]['signal3'],data[data['binnedliq_score']==4]['move_1D'], color = 'black')


# In[77]:


#least liquid, as R squared goes down


# In[491]:


x = data[data['binnedliq_score']==8][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedliq_score']==8]['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[492]:


sns.lineplot(x = data[data['binnedliq_score']==8]['signal3'], y= predictions)
sns.lineplot(x =  data[data['binnedliq_score']==8]['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedliq_score']==8]['signal3'],data[data['binnedliq_score']==8]['move_1D'], color = 'black')


# In[ ]:


#for loop that outputs coeff, confidence, overall fits, r-squared for each liquidity bucket


# In[135]:


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
    print('bin', i,': R-squared =',r_squared)
    print('Coefficients','\n',variable_coefs, '\n')
    print('Pvalues', '\n', p_values, '\n')
    print('C.I.', '\n', confidence_interval, '\n\n')


# combining LinReg into FOR loop to be able to see all important info obtained through LinReg model summary

# In[78]:


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
#     print(confidence_interval)
    print('-------------')
sub_df2=pd.concat(sub_df)
# pd.concat([pd.DataFrame(sub_df, columns=['coef','conf_a', 'conf_b', 'p_values', 'r_squared']),ignore_index=True)


# ### Maturity

# In[40]:


x = data[data['binnedmat_bucket']=='2-3 years'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedmat_bucket']=='2-3 years']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[43]:


sns.lineplot(x = data[data['binnedmat_bucket']=='2-3 years']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedmat_bucket']=='2-3 years']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedmat_bucket']=='2-3 years']['signal3'],data[data['binnedmat_bucket']=='2-3 years']['move_1D'], color = 'black')


# In[44]:


x = data[data['binnedmat_bucket']=='5 years'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedmat_bucket']=='5 years']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[45]:


sns.lineplot(x = data[data['binnedmat_bucket']=='5 years']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedmat_bucket']=='5 years']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedmat_bucket']=='5 years']['signal3'],data[data['binnedmat_bucket']=='5 years']['move_1D'], color = 'black')


# In[47]:


x = data[data['binnedmat_bucket']=='10 years'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedmat_bucket']=='10 years']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[48]:


sns.lineplot(x = data[data['binnedmat_bucket']=='10 years']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedmat_bucket']=='10 years']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedmat_bucket']=='10 years']['signal3'],data[data['binnedmat_bucket']=='10 years']['move_1D'], color = 'black')


# In[46]:


x = data[data['binnedmat_bucket']=='30 years'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedmat_bucket']=='30 years']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[49]:


sns.lineplot(x = data[data['binnedmat_bucket']=='30 years']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedmat_bucket']=='30 years']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedmat_bucket']=='30 years']['signal3'],data[data['binnedmat_bucket']=='30 years']['move_1D'], color = 'black')


# above: r-squared values increase as mat_bucket increases

# ### Quantity 

# In[58]:


x = data[data['binnedquantity_bonds']=='1-12'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedquantity_bonds']=='1-12']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[60]:


sns.lineplot(x = data[data['binnedquantity_bonds']=='1-12']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedquantity_bonds']=='1-12']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedquantity_bonds']=='1-12']['signal3'],data[data['binnedquantity_bonds']=='1-12']['move_1D'], color = 'black')


# In[63]:


x = data[data['binnedquantity_bonds']=='50-210'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedquantity_bonds']=='50-210']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[64]:


sns.lineplot(x = data[data['binnedquantity_bonds']=='50-210']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedquantity_bonds']=='50-210']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedquantity_bonds']=='50-210']['signal3'],data[data['binnedquantity_bonds']=='50-210']['move_1D'], color = 'black')


# In[65]:


x = data[data['binnedquantity_bonds']=='600-3000'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['binnedquantity_bonds']=='600-3000']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[66]:


sns.lineplot(x = data[data['binnedquantity_bonds']=='600-3000']['signal3'], y= predictions)
sns.lineplot(x = data[data['binnedquantity_bonds']=='600-3000']['signal3'], y=model.params[0], color = 'red')
plt.scatter( data[data['binnedquantity_bonds']=='600-3000']['signal3'],data[data['binnedquantity_bonds']=='600-3000']['move_1D'], color = 'black')


# see above: quantity seems to have little to no change in bins and therefore should not be a highly weighted factor in helping fine tune signal3 to predict move_1D

# # Sector

# In[72]:


x = data[data['industrySector']=='Consumer Cyclical'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Consumer Cyclical']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[73]:


x = data[data['industrySector']=='Basic Materials'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Basic Materials']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[75]:


x = data[data['industrySector']=='Technology'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Technology']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[76]:


x = data[data['industrySector']=='Industrial'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Industrial']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[93]:


x = data[data['industrySector']=='Financial'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Financial']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[94]:


x = data[data['industrySector']=='Consumer Non-cyclical'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Consumer Non-cyclical']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[95]:


x = data[data['industrySector']=='Utilities'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Utilities']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[96]:


x = data[data['industrySector']=='Energy'][['signal3', 'sig3equal1', 'sig3equalneg1']]
y = data[data['industrySector']=='Energy']['move_1D']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# # Playing around

# In[48]:


set.seed()


# In[ ]:


set.seed(2018)
quick_RF <- randomForest(x=all[1:1460,-79], y=all$SalePrice[1:1460], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]


# In[ ]:





# In[52]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)


# In[73]:





# In[87]:


from sklearn.linear_model import Lasso, LassoCV


x, y = (data[['binnedliq_score', 'bondlength', 'quantity_bonds','signal3', 'sig3equal1', 'sig3equalneg1']], data[['move_1D']])

model = Lasso().fit(x_train, y_train)
print(model)
score = model.score(x_train, y_train)
ypred = model.predict(x_test)
mse = mean_squared_error(y_test,ypred)
print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(model.alpha, score, mse, np.sqrt(mse)))

x_ax = range(len(ypred))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

# alphas = [-1,-0.50,0, 0.5, 1]
# lassocv = LassoCV(alphas=alphas, cv=5).fit(x,y)
# print(lassocv)
# score = lassocv.score(x,y)
# ypred = lassocv.predict(x_test)
# mse = mean_squared_error(y_test,ypred)
# print("Alpha:{0:.2f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(lassocv.alpha_, score, mse, np.sqrt(mse)))

# x_ax = range(len(x_test))
# plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show() 


# the values of the response variable to be categorical or discrete such as: “Yes” or “No”, “True” or “False”, 0 or 1. 

# In[106]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# Values for Predictor and Response variables
X =data[['binnedliq_score', 'bondlength', 'quantity_bonds','signal3', 'sig3equal1', 'sig3equalneg1']]
Y= data['move_1D']
# Create label encoder object
labels = preprocessing.LabelEncoder()
# Convert continous y values to categorical
Y_cat = labels.fit_transform(Y)
print(Y_cat)


# In[112]:


data['sig3equal1'].size


# In[107]:


# X =data[['binnedliq_score', 'bondlength', 'quantity_bonds','signal3', 'sig3equal1', 'sig3equalneg1']]
# Y= data['move_1D']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =  0.25,random_state = 0) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_cat)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


# In[ ]:




