#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# %%
data=pd.read_csv("spam.csv")
spam=[]
for i in data["Category"]:
    if i=="ham":
        spam.append(0)
    else:
        spam.append(1)
data["Spam"]=spam
del data["Category"]
print(data.head())

# %%
v=CountVectorizer()
features=v.fit_transform(data["Message"])
print(v.get_feature_names()) # set of all words in all training data
print(features.toarray()[0]) # array of size 8709 with each index having freq of corresponding word

# %%
xtn,xtt,ytn,ytt=train_test_split(data["Message"],data["Spam"],test_size=0.2)
train_freq=v.fit_transform(xtn)
mdl=MultinomialNB()
mdl.fit(train_freq,ytn)
print(mdl.score(v.transform(xtt),ytt))

# %%
email=["Win exciting offers by using this code:2wjc36"]
print(mdl.predict(v.transform(email)))

# %%
email=["Hi man. Long time no see. Use this code:2wjc36 to win a bike"]
print(mdl.predict(v.transform(email)))

# %%
