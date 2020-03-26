#%%
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import statistics as stat
import math
#%%
tdata=pd.read_csv("house.csv")
br=list(tdata["BR"])
br1=br.copy()
for i in range(len(br)):
    if math.isnan(br[i]):
        br[i]=0

while 0 in br:
    br.remove(0)

x=stat.median(br)
for i in range(len(br1)):
    if math.isnan(br1[i]):
        br1[i]=x
br=pd.Series(br1)
tdata["BR"]=br
print(tdata)
#%%
reg=linear_model.LinearRegression()
reg.fit(tdata[["Area","BR","Age"]],tdata["Price"])

# %%
print(reg.predict([[3000,4,30],[3200,3,18]]))

# %%
