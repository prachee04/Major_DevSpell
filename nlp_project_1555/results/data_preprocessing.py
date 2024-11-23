# Import necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('your_data.csv')

# Text cleaning function
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply text cleaning to 'text' column
df['text'] = df['text'].apply(clean_text)

# Tokenization function
def tokenize_text(text):
    return word_tokenize(text)

# Apply tokenization to 'text' column
df['text'] = df['text'].apply(tokenize_text)

# Stop word removal function
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word not in stop_words]

# Apply stop word removal to 'text' column
df['text'] = df['text'].apply(remove_stop_words)

# Convert list of tokens back to string
df['text'] = df['text'].apply(lambda x: ' '.join(x))

# Split data into training and testing sets
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)