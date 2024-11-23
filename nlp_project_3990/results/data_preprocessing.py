# Import necessary libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import re

# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Define a function for text cleaning
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a string
    text = ' '.join(tokens)
    
    return text

# Apply the text cleaning function to the 'Movie_Title' column
df['Movie_Title'] = df['Movie_Title'].apply(clean_text)

# Define a function for text vectorization
def vectorize_text(text):
    # Initialize a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer to the text data and transform it into vectors
    vectors = vectorizer.fit_transform(text)
    
    return vectors

# Apply the text vectorization function to the 'Movie_Title' column
vectors = vectorize_text(df['Movie_Title'])

# Split the data into training and testing sets
train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, df['main_genre'], test_size=0.2, random_state=42)