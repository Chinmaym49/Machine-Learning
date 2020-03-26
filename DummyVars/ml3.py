#%%
import numpy as np
import pandas as pd
from sklearn import linear_model,model_selection

# %%
data=pd.read_csv("homeprices.csv")
print(data)

# %%
s=list(set(data["town"]))
dumvars=pd.get_dummies(data["town"])
print(dumvars)
for twn in s:
    data[twn]=dumvars[twn]
del data["town"]
del data[s[-1]]
print(data)
# %%
x_train,x_test,y_train,y_test=model_selection.train_test_split(np.array(data[["area","monroe township","robinsville"]]),np.array(data["price"]),test_size=0.1)
print(x_train,y_train)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.predict(x_test))

# %%
