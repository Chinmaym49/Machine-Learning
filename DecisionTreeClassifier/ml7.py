#%%
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#%%
data=pd.read_csv("salaries.csv")
print(data.head())

# %%
le=LabelEncoder()
data["comp"]=le.fit_transform(data["company"])
data["jb"]=le.fit_transform(data["job"])
data["deg"]=le.fit_transform(data["degree"])
del data["company"]
del data["job"]
del data["degree"]
print(data.head())

# %%
model=tree.DecisionTreeClassifier()
model.fit(data[["comp","jb","deg"]],data["salary_more_then_100k"])
print(model.predict([[2,1,1]]))

# %%
