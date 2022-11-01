import pandas as pd
import seaborn as sns
from preprocess import DataProcessor
from config import PreprocessModel
from sklearn.preprocessing import StandardScaler

data_frame = sns.load_dataset("iris")

pm = PreprocessModel()
pm.remove_outliers = True
pm.scale = True
pm.scaler = StandardScaler
pm.label_encode = True

dp = DataProcessor(data_frame, pm, "species")
# dp.summary()
df = dp.process_data()
print(data_frame.head())
print(dp.process_as_previous(data_frame.sample(2)).head())
