import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
import statistics as s

df=pd.read_csv("C:/Users/Kiran/Downloads/heart.csv")
print(df.head())

print(df.info())

print("size of dataset ",df.size)

print("shape of dataset = ",df.shape)

print(df.describe())

print("data types for each column\n",df.dtypes.value_counts())

n=df.columns[df.dtypes=='object']
print(df[n].isnull().sum())

print(df[n].isnull().sum().sort_values(ascending=False)/len(df))

print("mean=",s.mean(df['Age']))

print(df["Sex"])

from sklearn.model_selection import train_test_split
col_delete=['HeartDisease','Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
x=df.drop(col_delete,axis=1)
y=df.HeartDisease
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
print("train dataset\n",xtest)
print("\n\nTest dataset\n",ytest)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Fit the model to the training data
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)




from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score

print(model.classes_)

cm=confusion_matrix(ytest,y_pred,labels=model.classes_)
pl=ConfusionMatrixDisplay(cm,display_labels=['Diseased','Not Diseased'])
pl.plot()
mp.show()

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)
y_predict=knn.predict(xtest)
ks=accuracy_score(ytest,y_predict)
print(ks)


from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))



