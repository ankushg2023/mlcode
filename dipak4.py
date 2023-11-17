import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Mall_Customers.csv')
df1=df.drop('Genre',axis=1)
corr=df1.corr()
sns.heatmap(corr,fmt='.1f',annot=True)
plt.show()
plt.figure(figsize=(10,10))
sns.barplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)
plt.show()
plt.scatter(df['Age'],df['Annual Income (k$)'])
plt.show()
from sklearn.cluster import KMeans
x=df.iloc[:,[2,3,4]].values
sse=[]
for i in range(1,30):
    km=KMeans(n_clusters=i)
    km.fit(x)
    sse.append(km.inertia_)

print(sse)
plt.plot(range(1,30),sse)
plt.show()
km=KMeans(n_clusters=5)
y=km.fit_predict(x)
print(y)
df['cluster']=y
print(df.head())
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
scaler.fit(df[['Income']])
df['Income']=scaler.transform(df[['Income']])
df2=df[df.cluster==0]
df3=df[df.cluster==1]
df4=df[df.cluster==2]
df5=df[df.cluster==3]
df6=df[df.cluster==4]

plt.scatter(df2.Age,df2.Income,color='red')
plt.scatter(df3.Age,df3.Income,color='blue')
plt.scatter(df4.Age,df4.Income,color='green')
plt.scatter(df5.Age,df5.Income,color='yellow')
plt.scatter(df6.Age,df6.Income,color='black')
plt.show()