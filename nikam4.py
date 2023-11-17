import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Programming/Python/Mall_Customers.csv")

from sklearn.model_selection import train_test_split,cross_val_score

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
df['cluster']=y;
print(df.head())

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
scaler.fit(df[['Income']])
df['Income'] = scaler.fit_transform(df[['Income']])
df2=df[df.cluster==0]
df3=df[df.cluster==1]
df4=df[df.cluster==2]
df5=df[df.cluster==3]
df6=df[df.cluster==4]

plt.scatter(df2.Age,df2.Income,color='red')
plt.scatter(df3.Age,df3.Income,color='blue')
plt.scatter(df4.Age,df4.Income,color='yellow')
plt.scatter(df5.Age,df5.Income,color='green')
plt.scatter(df6.Age,df6.Income,color='purple')
plt.show()

