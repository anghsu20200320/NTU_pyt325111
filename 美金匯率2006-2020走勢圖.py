import pandas as pd

df = pd.read_csv(r"C:\Users\USER\Desktop\2006-2020美金走勢\資料來源\20060529 usd.csv",index_col='date',parse_dates=['date'])

#index換成日期，這樣之後在查找時，會比較方便
#其中index_col就是將Date這條column當作是index，而parse_dates可以將Date轉換成程式瞭解的日期格式，而非單純的字串。

print(df.count())
print(df.head(10))


#import matplotlib 
import matplotlib.pyplot as plt
#df.plt.plot()
#plt.show()

#%pylab inline
df.plot(kind='line',y='price')
plt.show()

#其中的squeeze就是將dataframe變成series的function
print(df.count())
print(df.head(10))
#print(price.head(10))



adj_price=df.price.values
x=adj_price.reshape(-1, 1)
y=adj_price.reshape(-1, 1)
#plt.scatter(x,y,alpha=0.5)
#plt.show()

from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
#from sklearn import datasets


x_train=x[:-854]
x_test=x[-854:]
y_train=y[:-854]
y_test=y[-854:]



regr=LinearRegression()
regr.fit(x_train,y_train)


plt.plot(y_test)
plt.plot(regr.predict(x_test))
plt.show()

