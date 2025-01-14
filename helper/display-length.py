import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/imdb_clean.csv')

df['length'] = df['review'].apply(lambda x: len(x.split()))

# drop ones with length >500
df = df[df['length'] < 200]
df.drop(columns=['length'], inplace=True)

df.to_csv('data/imdb_clean_short.csv', index=False)