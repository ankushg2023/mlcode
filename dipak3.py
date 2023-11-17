import pandas as pd
import seaborn as sns
df=pd.read_csv('Admission_Predict.csv')

df.head()
df.columns
df.shape

from sklearn.preprocessing import Binarizer
bi=Binarizer(threshold=0.75)
df['Chance of Admit ']=bi.fit_transform(df[['Chance of Admit ']])

x=df.drop('Chance of Admit ',axis=1)
y=df['Chance of Admit ']

x
y=y.astype('int')
y
sns.countplot(x = y);
y.value_counts()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25)

x_train.shape
x_test.shape

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)

classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)

result = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
    })

result

# aaplyala jr hyachi accuracy bghaychi asn 

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

new = [[]]
classifier.predict(new)[0]

#from sklearn.tree import plot_tree
#plot_tree(classifier,fontsize)