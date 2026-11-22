import pandas as pd

df = pd.read_csv("dataset/News.csv", on_bad_lines="skip")
print("Rows loaded:", len(df))
print(df.head(5).to_string())
print()
print(df["category"].value_counts().head(20))