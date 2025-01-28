#!/usr/bin/env python
# coding: utf-8

# In[2]:


#### TASK 1 EDA and Business Insights
import pandas as pd

# Load datasets
customers = pd.read_csv('Customers 1.csv')
products = pd.read_csv('Products 1.csv')
transactions = pd.read_csv('Transactions 1.csv')

# Display basic info
print(customers.info())
print(products.info())
print(transactions.info())


# In[7]:


customers.head()


# In[8]:


products.head()


# In[9]:


transactions.head()


# In[10]:


print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())


# In[11]:


transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

region_counts = customers['Region'].value_counts()
sns.barplot(x=region_counts.index, y=region_counts.values)
plt.title('Customers by Region')
plt.show()


# In[13]:


top_products = transactions.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False)
print(top_products.head())


# In[14]:


merged_data = pd.merge(transactions, products, on='ProductID')
category_revenue = merged_data.groupby('Category')['TotalValue'].sum()
print(category_revenue)


# In[15]:


transactions.groupby(transactions['TransactionDate'].dt.month)['TotalValue'].sum().plot()
plt.title('Monthly Revenue Trend')
plt.show()


# In[16]:


###TASK 2 - LOOKALIKE MODEL


# In[17]:


#AGGREGATE CUSTOMER DATA

customer_summary = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum'
}).reset_index()


# In[18]:


#NORMALIZE DATA

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
customer_summary[['TotalValue', 'Quantity']] = scaler.fit_transform(customer_summary[['TotalValue', 'Quantity']])


# In[19]:


from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=4, metric='cosine')  # Include the customer itself
knn.fit(customer_summary[['TotalValue', 'Quantity']])
distances, indices = knn.kneighbors(customer_summary[['TotalValue', 'Quantity']])


# In[20]:


lookalikes = {}
for i, customer_id in enumerate(customer_summary['CustomerID'][:20]):
    similar_customers = [(customer_summary['CustomerID'][indices[i][j]], distances[i][j]) for j in range(1, 4)]
    lookalikes[customer_id] = similar_customers

print(lookalikes)


# In[21]:


import csv

with open('Lookalike.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['CustomerID', 'LookalikeCustomers'])
    for key, value in lookalikes.items():
        writer.writerow([key, value])


# In[22]:


### TASK 3 - CUSTOMER SEGMENTATION


# In[34]:


customer_data = pd.merge(customers, customer_summary, on='CustomerID')


# In[30]:


import os
os.environ["OMP_NUM_THREADS"] = "1"


# In[32]:


#RUN KMEANS CLUSTRING

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['TotalValue', 'Quantity']])




# In[33]:


import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


# In[25]:


#CALCULATE DB INDEX

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(customer_data[['TotalValue', 'Quantity']], customer_data['Cluster'])
print(f'DB Index: {db_index}')


# In[26]:


#VISUALIZE CLUSTERS

sns.scatterplot(x='TotalValue', y='Quantity', hue='Cluster', data=customer_data, palette='Set1')
plt.title('Customer Clusters')
plt.show()


# In[ ]:




