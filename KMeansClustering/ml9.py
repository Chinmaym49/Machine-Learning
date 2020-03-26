#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %%
tdata=pd.read_csv("income.csv")
del tdata["Name"]
print(tdata.head())

# %%
sc=MinMaxScaler()
sc.fit(tdata[["Age"]])
tdata["Age"]=sc.transform(tdata[["Age"]])
sc.fit(tdata[["Income($)"]])
tdata["Income($)"]=sc.transform(tdata[["Income($)"]])
print(tdata.head())

# %%
plt.xlabel("Age")
plt.ylabel("Income")
plt.scatter(tdata["Age"],tdata["Income($)"])

# %%
sse=[]
for k in range(1,len(tdata)):
    mdl=KMeans(n_clusters=k)
    mdl.fit(tdata[["Age","Income($)"]])
    sse.append(mdl.inertia_)
print(sse)

# %%
plt.xlabel("K")
plt.ylabel("SSE")
plt.plot(list(range(1,len(tdata))),sse)
plt.show()

# %%
model=KMeans(n_clusters=3)
print(model.fit_predict(tdata[["Age","Income($)"]]))

# %%
