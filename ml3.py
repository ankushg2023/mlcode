import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("C:/Users/Kiran/Downloads/winequality-red.csv")
print(df)


y = df['coa']
x = df.drop('coa', axis=1)
corr=df.corr()
sns.heatmap(corr,annot=True,fmt='.1f')
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
y_pred=model.predict(xtest)

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
cm=confusion_matrix(ytest,y_pred)
pl=ConfusionMatrixDisplay(cm)
pl.plot()
plt.show()

print("Accuracy of our Model=",accuracy_score(ytest,y_pred))