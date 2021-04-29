import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('salary.csv')


X = dataset.iloc[:,:3]
y = dataset.iloc[:,-1]

regression = LinearRegression()
regression.fit(X,y)
pickle.dump(regression, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,3,7]]))