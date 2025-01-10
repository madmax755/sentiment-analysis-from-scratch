import pandas as pd # type: ignore
import re

df = pd.read_csv("data/imdb_raw.csv")

def clean_text(text):
    # handle NaN/None values
    if pd.isna(text):
        return ""
    
    # ensure text is string
    text = str(text)
    
    # convert to lowercase
    text = text.lower()
    
    # remove html tags (fixed regex)
    text = re.sub(r'<.*?>', '', text)
    
    # remove special characters but keep apostrophes
    text = re.sub(r'[^a-z\'\s]', ' ', text)
    
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_sentiment(sentiment):
    if sentiment == "positive":
        return 1.0
    elif sentiment == "negative":
        return -1.0
    else:
        return None

# clean data
df["review"] = df["review"].apply(clean_text)
df["sentiment"] = df["sentiment"].apply(clean_sentiment)

# split into training and test data
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# save to csv
train_df.to_csv("data/imdb_clean_train.csv")
test_df.to_csv("data/imdb_clean_test.csv")
