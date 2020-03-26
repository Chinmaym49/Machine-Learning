#%%
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
#%%
df=pd.read_csv("property.csv")
# print(df)

plt.xlabel("Size")
plt.ylabel("Price")
plt.scatter(df.Size,df.Price,color='red')

print(type(df.Size))
print(df["Size"]) # Same as above(Series)
print(type(df[["Size"]])) # DF
reg=linear_model.LinearRegression()
reg.fit(df[["Size"]],df.Price) # takes DF of ip and series of op


lis=[1,2,3,4,5,6,7,8,9,10]
d=pd.DataFrame({'Size':lis})
print(reg.predict(d)) # takes a DF or a 2D array(predict([[6],[7],[8]]))


# %%
plt.scatter(df.Size,df.Price,color='red')
plt.plot(d,reg.predict(d))

# %%
