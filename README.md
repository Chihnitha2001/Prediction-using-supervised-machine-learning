# Prediction-using-supervised-machine-learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

filename="http://bit.ly/w-data"
data = pd.read_csv(filename)
print("data imported successfully")
print(data.head())
X_train = np.array(data['Hours'])
y_train = np.array(data['Scores'])
plt.scatter(X_train,y_train)
plt.title("HoursVsPercentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.show()
classifier=LinearRegression()
classifier=LinearRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_train)
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred,color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.show()
