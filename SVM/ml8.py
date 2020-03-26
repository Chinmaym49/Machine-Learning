#%%
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
irdata=load_iris()
print(dir(irdata))
print(irdata.data[65])
print(irdata.feature_names)
print(irdata.target[65])
print(irdata.target_names)

# %%
plt.xlabel("sepal length (cm)")
plt.ylabel("type")
spdata=[]
for i in irdata.data:
    spdata.append(i[0])
plt.scatter(spdata,irdata.target)
# %%
model=SVC()
xtn,xtt,ytn,ytt=train_test_split(irdata.data,irdata.target,test_size=0.2)
model.fit(xtn,ytn)
print(model.score(xtt,ytt))

# %%
