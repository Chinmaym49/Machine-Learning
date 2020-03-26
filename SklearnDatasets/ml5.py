#%%
import numpy as np
import pandas as pd
from sklearn import linear_model,model_selection
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# %%
dig=load_digits()
print(dir(dig))
print(len(dig.data))
print(dig.target_names)
print(dig.data[73])
plt.matshow(dig.images[73])
print(dig.target[73])

# %%
reg=linear_model.LogisticRegression()
xtn,xtt,ytn,ytt=model_selection.train_test_split(dig.data,dig.target,test_size=0.2)
reg.fit(xtn,ytn)

# %%
print(reg.predict(xtt))
print(reg.score(xtt,ytt))
print(dig.target[100])
plt.matshow(dig.images[100])
print(reg.predict([dig.data[100]]))

# %%
