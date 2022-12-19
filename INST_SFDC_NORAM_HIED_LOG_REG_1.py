#!/usr/bin/env python
# coding: utf-8

# In[84]:


#import packages 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

#import more packages
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[85]:


#load data
df = pd.read_csv('NORAM_HIED_NEW+UP_LOG_REG_1.csv',encoding='latin-1')


# In[86]:


df.info()


# In[87]:


#identify null variables
df.isnull().sum()


# In[88]:


df.hist()


# In[89]:


#drop complicating variables
df= df.drop(['Number of Emails Before Response'], axis=1)
df= df.drop(['Gap between Close Date and Go Live Date'], axis=1)
df= df.drop(['Lead Source'], axis=1)
df= df.drop(['ï»¿Account Name'], axis=1)
df= df.drop(['Opportunity Name'], axis=1)
df= df.drop(['Account ID 18'], axis=1)
df= df.drop(['Opportunity ID 18'], axis=1)
df= df.drop(['Contact: Title'], axis=1)
df= df.drop(['Implementation Package'], axis=1)
df= df.drop(['Primary LMS'], axis=1)
df= df.drop(['Primary SIS'], axis=1)
df= df.drop(['Org Type'], axis=1)
df= df.drop(['Customer Success Region'], axis=1)


# In[90]:


df.info()


# In[91]:


import missingno as msno
msno.matrix(df)


# In[92]:


#verify there are no null values
df.isnull().sum()


# In[93]:


#verify the distribution after imputation
df.hist()


# In[94]:


df.info()


# In[95]:


#perform imputation for nulls (If there is symmetry in the dataset, use the mean. If the dataset is skewed use the median. Use the mode with qualitative variables.)
df['Account FTE/Users'].fillna(df['Account FTE/Users'].mean(), inplace=True)
df['Department Source'].fillna(df['Department Source'].mode()[0], inplace=True)
df['Product Type'].fillna(df['Product Type'].mode()[0], inplace=True)
df['Inbound/Outbound Source'].fillna(df['Inbound/Outbound Source'].mode()[0], inplace=True)
df['Days Since Last Activity'].fillna(df['Days Since Last Activity'].mean(), inplace=True)
df['FFT Teachers Active'].fillna(df['FFT Teachers Active'].median(), inplace=True)
df['FFT Students Active'].fillna(df['FFT Students Active'].median(), inplace=True)
df['FFT Observers Active'].fillna(df['FFT Observers Active'].median(), inplace=True)
df['FFT Courses Active'].fillna(df['FFT Courses Active'].median(), inplace=True)


# In[96]:


import missingno as msno
msno.matrix(df)


# In[97]:


#verify there are no null values
df.isnull().sum()


# In[98]:


#verify the distribution after imputation
df.hist()


# In[99]:


df.dtypes


# In[100]:


# create dummy variables 
dummies= pd.get_dummies(df[["Fiscal Quarter","Inbound/Outbound Source","Type","Department Source","Product Type"]])
df = pd.concat([df, dummies], axis=1)


# In[101]:


df.describe()


# In[102]:


df = df.drop(['Fiscal Quarter'], axis=1)
df = df.drop(['Inbound/Outbound Source'], axis=1)
df = df.drop(['Type'], axis=1)
df = df.drop(['Department Source'], axis=1)
df = df.drop(['Product Type'], axis=1)


# In[104]:


df.dtypes


# In[105]:


df.info()


# In[107]:


#drop k-1 dummy variables
df = df.drop(['Fiscal Quarter_Q1'], axis=1)
df = df.drop(['Inbound/Outbound Source_Inbound'], axis=1)
df = df.drop(['Department Source_Partner Generated'], axis=1)
df = df.drop(['Product Type_Subscription Add-ons'], axis=1)
df = df.drop(['Opp Description Not Blank'], axis=1)
df = df.drop(['Type_Upsell'], axis=1)


# In[108]:


df = df.drop(['Product Type_Subscription Add-ons'], axis=1)


# In[109]:


df = df.drop(['Opp Description Not Blank'], axis=1)


# In[110]:


df = df.drop(['Type_Upsell'], axis=1)


# In[112]:


# transform variables
df['Fiscal Quarter_Q2'] = df['Fiscal Quarter_Q2'].astype(float)
df['Fiscal Quarter_Q3'] = df['Fiscal Quarter_Q3'].astype(float)
df['Fiscal Quarter_Q4'] = df['Fiscal Quarter_Q4'].astype(float)
df['Inbound/Outbound Source_Outbound'] = df['Inbound/Outbound Source_Outbound'].astype(float)
df['Department Source_CS Generated'] = df['Department Source_CS Generated'].astype(float)
df['Department Source_CX Generated'] = df['Department Source_CX Generated'].astype(float)
df['Department Source_Channel Generated'] = df['Department Source_Channel Generated'].astype(float)
df['Department Source_Marketing Generated'] = df['Department Source_Marketing Generated'].astype(float)
df['Department Source_Sales Generated'] = df['Department Source_Sales Generated'].astype(float)
df['Product Type_Cloud SaaS Subscription'] = df['Product Type_Cloud SaaS Subscription'].astype(float)
df['Product Type_Custom Development'] = df['Product Type_Custom Development'].astype(float)
df['Product Type_Implementation'] = df['Product Type_Implementation'].astype(float)
df['Product Type_Pilot'] = df['Product Type_Pilot'].astype(float)
df['Product Type_Support'] = df['Product Type_Support'].astype(float)
df['Product Type_Training'] = df['Product Type_Training'].astype(float)


# In[117]:


df.info()


# In[114]:


df= df.drop(['Org Type'], axis=1)
df= df.drop(['Customer Success Region'], axis=1)


# In[116]:


# Calculate the required sample size
N = 10 k / p
1,193=(10*62) / .51943


# In[ ]:


# Calculate the required sample size
N = 10 k / p
1,307=(10*62) / .4742


# In[118]:


#bivariate visualization 
pd.plotting.scatter_matrix(df, alpha = 1, figsize = (30, 20))

plt.show()


# In[119]:


#import packages for log regression
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[120]:


#perform logistic regression
x= df.drop('Stage (1=Closed Won, 0=Closed Lost)', axis =1)
y= df['Stage (1=Closed Won, 0=Closed Lost)']


# In[121]:


model = LogisticRegression(solver= 'liblinear' , random_state=0)

model.fit(x,y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn' , n_jobs=None, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


# In[122]:


# create a OLS model
model = sm.OLS(y, x)

# fit the data
est = model.fit()

# show summary
est.summary()


# In[130]:


#create new data frame with statistically significant pvalues
df = df[['FTE <1,000',
        'FTE 1,000-5,000',
        'FTE 5,000-10,000',
        'FTE >10,000',
        'Sales Owner: Regional Director',
        'Sales Owner: Account Executive',
        'Age',
        'Lead Source: Chat',
        'Lead Source: Cold Call',
        'Lead Source: CSM Generated',
        'Lead Source: Tradeshow',
        'Lead Source: Free_for_teacher',
        'Lead Source: "Demo"',
        'Lead Source: "Web"',
        'Contracted Qty <1,000',
        'Contracted Qty 1,000-5,000',
        'Contracted Qty 5,000-10,000',
        'Contracted Qty >10,000',
        'Contract Qty = Account FTE',
        'Opp Description Inludes "Price/Pricing"',
        'Primary SIS Not Blank',
        'Primary LMS: Blackboard',
        'Primary LMS: D2L',
        'Primary LMS: Moodle',
        'Primary LMS: None/Blank',
        'Days Since Last Activity',
        'FFT Teachers Active', 
        'FFT Courses Active',
        'Fiscal Quarter_Q3', 
        'Fiscal Quarter_Q4',    
        'Inbound/Outbound Source_Outbound',     
        'Type_New Logo', 
        'Department Source_CS Generated', 
        'Department Source_Marketing Generated', 
        'Product Type_Cloud SaaS Subscription',
        'Product Type_Custom Development',
        'Product Type_Implementation',
        'Product Type_Migration',
        'Product Type_Partnership',
        'Product Type_Strategic Services',
        'Product Type_Training',
        'Stage (1=Closed Won, 0=Closed Lost)',]]


# In[131]:


# test for multicolenearity 
corr = df.corr()
display(corr)

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')


# In[132]:


# find VIF  
df_before = df

X1 = sm.tools.add_constant(df_before)

series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)

print('DATA BEFORE')
print('-'*100)
display(series_before)


# In[133]:


#drop Age
df_after = df.drop(['Contracted Qty <1,000'], axis=1)

X2 = sm.tools.add_constant(df_after)

series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

print('DATA AFTER')
print('-'*100)
display(series_after)


# In[134]:


#drop Department Source_Sales Generated
df_after = df_after.drop(['Product Type_Strategic Services'], axis=1)

X2 = sm.tools.add_constant(df_after)

series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

print('DATA AFTER')
print('-'*100)
display(series_after)


# In[135]:


#drop Department Source_Sales Generated
df_after = df_after.drop(['Lead Source: Cold Call'], axis=1)

X2 = sm.tools.add_constant(df_after)

series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

print('DATA AFTER')
print('-'*100)
display(series_after)


# In[136]:


#drop Department Source_Sales Generated
df_after = df_after.drop(['Primary LMS: D2L'], axis=1)

X2 = sm.tools.add_constant(df_after)

series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

print('DATA AFTER')
print('-'*100)
display(series_after)


# In[137]:


#import packages
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[138]:


#perform logistic regression
x= df_after.drop('Stage (1=Closed Won, 0=Closed Lost)', axis =1)
y= df_after['Stage (1=Closed Won, 0=Closed Lost)']


# In[139]:


model = LogisticRegression(solver= 'liblinear' , random_state=0)

model.fit(x,y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn' , n_jobs=None, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# create a OLS model
model = sm.OLS(y, x)

# fit the data
est = model.fit()

# show summary
est.summary()


# In[ ]:




