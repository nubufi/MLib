import pandas as pd
from preprocess import DataProcessor
from config import PreprocessModel, ClassificationModel
from classification import Classification

data_frame = pd.read_csv("datasets/diabetes.csv")
pm = PreprocessModel()
dp = DataProcessor(data_frame, pm, "Outcome")
df = dp.process_data()
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

cm = ClassificationModel()
clf = Classification(x, y, cm)
model = clf.create_model()
