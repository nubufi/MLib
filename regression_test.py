import pandas as pd
from preprocess import DataProcessor
from config import RegressionModel, PreprocessModel
from regression import Regression
from sklearn.metrics import r2_score

df = pd.read_csv("datasets/gmpe1.csv")
dp = DataProcessor(df, PreprocessModel(), "PGA")
data_frame = dp.process_data()
x_train, x_test, y_train, y_test = dp.split_data()

rm = Regression(x_train, y_train, RegressionModel())
model = rm.create_model()

y_predict = model.predict(x_test)
test_score = r2_score(y_test, y_predict)
print("test_score = ", test_score)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_predict)
plt.plot([-1.5, 1.7], [-1.5, 1.7], color='red')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
plt.close()
