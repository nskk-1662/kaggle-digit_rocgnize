import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("sample_submission.csv")

x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
