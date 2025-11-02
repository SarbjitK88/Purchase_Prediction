# %% [markdown]
# Black Friday Dataset EDA And Feature Engineering

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Problem Statement
# ##### A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# ##### Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# %%
#importing the dataset
df_train=pd.read_csv('blackFriday_train.csv')
df_train.head()

# %%
##  import the test data
df_test=pd.read_csv('blackFriday_test.csv')
df_test.head()

# %%
##MErge both train and test data
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
df.head(5)

# %%
##Basic 
df.info()

# %%
df.describe()

# %%
df.drop(['User_ID'],axis=1,inplace=True)

# %%
df.head()

# %%
##HAndling categorical feature Gender
df['Gender']=df['Gender'].map({'F':0,'M':1})
df.head()

# %%
## Handle categorical feature Age
df['Age'].unique()

# %%
#pd.get_dummies(df['Age'],drop_first=True)
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})

# %%
##second technqiue
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['Age']= label_encoder.fit_transform(df['Age'])
 
df['Age'].unique()

# %%
df.head()

# %%
##fixing categorical City_categort
df_city=pd.get_dummies(df['City_Category'],drop_first=True)

# %%
df_city.head()

# %%
df=pd.concat([df,df_city],axis=1)
df.head()

# %%
##drop City Category Feature
df.drop('City_Category',axis=1,inplace=True)

# %%

## Missing Values
df.isnull().sum()

# %%
## Focus on replacing missing values
df['Product_Category_2'].unique()

# %%
df['Product_Category_2'].value_counts()

# %%

df['Product_Category_2'].mode()[0]

# %%
## Replace the missing values with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])

# %%
df['Product_Category_2'].isnull().sum()

# %%
## Product_category 3 replace missing values
df['Product_Category_3'].unique()

# %%
df['Product_Category_3'].value_counts()

# %%

## Replace the missing values with mode
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])

# %%
df.head()

# %%
df.shape

# %%

df['Stay_In_Current_City_Years'].unique()

# %%
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')

# %%

df.head()

# %%
df.info()

# %%
##convert object into integers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df.info()

# %%
df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)

# %%
df.info()

# %%
##Visualisation Age vs Purchased
sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)

# %% [markdown]
# ### Purchasing of men is high then women

# %%

## Visualization of Purchase with occupation
sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# %%
df.head()

# %% [markdown]
# ## Feature Scaling 

# %%

df_test=df[df['Purchase'].isnull()]

# %%
df_train=df[~df['Purchase'].isnull()]

# %%
X=df_train.drop('Purchase',axis=1)

# %%
X.head()

# %%
X.shape

# %%
y=df_train['Purchase']

# %%
y.shape

# %%
y

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.30, random_state=42)

# %%
X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)

# %% [markdown]
# ### Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# %%





# %%


# %%


# %%


# %%



