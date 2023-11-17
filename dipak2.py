import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('temperatures.csv')
df.head()
df.shape
df.columns

x = df['YEAR']
y = df['ANNUAL']


plt.title('India Annual temp')
plt.xlabel('Year')
plt.ylabel('Annual temp')
plt.scatter(x,y)
x.shape
# x la 2 dimensional kel array mdhi represent kela tyala
x=x.values
x=x.reshape(117,1)
x.shape
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()

regressor.coef_

regressor.intercept_

regressor.predict([[2050]])

predicted = regressor.predict(x)
#absolute error
abs(y-predicted)
import numpy as np
#mean absolute error
np.mean(abs(y-predicted))

#direct library import krun pn aapn mean absolute error kadhu shkto

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,predicted)

from sklearn.metrics import mean_squared_error
mean_squared_error(y,predicted)

#r ssqured matrices ha error show krt asto 

from sklearn.metrics import r2_score
r2_score(y,predicted)

regressor.score(x, y)

#ata graphical mdhi represent kraycha

plt.title('Regresssion model')
plt.xlabel('Year')
plt.ylabel('Annual Temp')
plt.scatter(x,y,label='actual',color='r')
plt.plot(x,predicted,label='predicted',color='g')
plt.legend()

sns.regplot(x='YEAR',y='ANNUAL',data=df)


