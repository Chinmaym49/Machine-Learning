#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn import model_selection
import numpy as np
import pandas as pd

# %%
bc=load_breast_cancer()
print(dir(bc))
print(len(bc.data))
print(bc.data[68])
print(bc.target_names)
print(bc.feature_names)
print(bc.target[68])

# %%
mod=KNeighborsClassifier(n_neighbors=7)
xtn,xtt,ytn,ytt=model_selection.train_test_split(bc.data,bc.target,test_size=0.2)
mod.fit(xtn,ytn)
print(mod.predict(xtt))
print(mod.score(xtt,ytt))

# %%
