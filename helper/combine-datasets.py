import pandas as pd

df1 = pd.read_csv("data/imdb_clean_train.csv")
df2 = pd.read_csv("data/twitter_clean_train.csv")

df = pd.concat([df1, df2])
df = df.sample(frac=1).reset_index(drop=True)

df = df[["text", "sentiment"]]

df.to_csv("data/combined_train.csv")

df1 = pd.read_csv("data/imdb_clean_test.csv")
df2 = pd.read_csv("data/twitter_clean_test.csv")
df = pd.concat([df1, df2])
df = df.sample(frac=1).reset_index(drop=True)

df = df[["text", "sentiment"]]
df.to_csv("data/combined_test.csv")