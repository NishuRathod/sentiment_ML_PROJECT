
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#", '', text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    # Lowercase and tokenize
    text = text.lower().split()
    # Remove stopwords and lemmatize
    text = [lemmatizer.lemmatize(word) for word in text if word not in STOPWORDS]
    return " ".join(text)
#os.makedirs('data/processed',exist_ok=True)
def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, encoding='latin1', header=None)
    df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['sentiment'] = df['target'].apply(lambda x: 0 if x == 0 else 1)  # Binary sentiment
    df = df[['cleaned_text', 'sentiment']]
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/archive/noemoticon.csv", "data/processed/cleaned_data.csv")

    
