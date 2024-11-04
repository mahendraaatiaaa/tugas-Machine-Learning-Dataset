import pandas as pd

# Memuat dataset IRIS dari file CSV
df_iris = pd.read_csv("Iris.csv")

# Menyimpan DataFrame ke file Excel
df_iris.to_excel("iris_dataset.xlsx", index=False)  # index=False untuk tidak menyertakan index
