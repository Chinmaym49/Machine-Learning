#%%
import numpy as np
import pandas as pd
from sklearn import linear_model,model_selection
import matplotlib.pyplot as plt

# %%
data=pd.read_csv("insurance_data.csv")
print(data)

# %%
plt.xlabel("Age")
plt.ylabel("Bought or not")
plt.scatter(data["age"],data["bought_insurance"])
plt.show()

# %%
x_train,x_test,y_train,y_test=model_selection.train_test_split(data[["age"]],data["bought_insurance"],test_size=0.2)


# %%
reg=linear_model.LogisticRegression()
reg.fit(x_train,y_train)


# %%
print(x_test)
print(reg.predict(x_test))

# %%
