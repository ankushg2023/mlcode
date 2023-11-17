import pandas as pd 
df=pd.read_csv('Heart.csv')
df.head()
df.shape
df.isnull()
df.isnull().sum()
df.count()
df.dtypes
df == 0
df [df == 0]
df[df==0].count()
df.columns
df['Age'].mean()
df['Age'].mode()
df['Age'].median()
df[['Age','Sex','ChestPain','RestBP','Chol']]
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, random_state=0,test_size=0.25)
train.shape