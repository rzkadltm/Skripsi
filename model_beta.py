from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from kbbi import KBBI
from nltk.tokenize import word_tokenize
import pandas as pd
import string
import nltk
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Read the CSV file
data_frame = pd.read_csv('data_karyatulisilmiah.csv')

# Select relevant columns
data = data_frame[['Tanggal', 'Judul', 'Pengarang', 'Abstrak','URL']]

# Remove rows with missing values
data = data.dropna()

# Combine title and abstract into a single column
data['Judul_Pengarang_Abstrak'] = data[['Judul', 'Pengarang', 'Abstrak']].agg(' '.join, axis=1)

# Apply preprocessing to text columns
data['Judul_Pengarang_Abstrak'] = data['Judul_Pengarang_Abstrak'].apply(preprocess_text)

# Initialize TfidfVectorizer with adjusted parameters
vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)

# def recommendations_beta(search):
#     # Preprocess the search query
#     contents = preprocess_text(search)
    
#     # Transform text using TfidfVectorizer
#     keywords = vectorizer.fit_transform(data.Judul_Pengarang_Abstrak)
#     code = vectorizer.transform([contents])
    
#     # Calculate cosine similarity
#     dist = cosine_similarity(code, keywords)
    
#     # Get top 10 most similar articles
#     top_indices = dist.argsort()[0, :-21:-1]
#     result = data.iloc[top_indices]
#     result1 = result[['Tanggal', 'Judul', 'Pengarang', 'Abstrak']]
    
#     return result1

def recommendations_beta(search):
    # Transform text using TfidfVectorizer
    keywords = vectorizer.fit_transform(data.Judul_Pengarang_Abstrak)
    contents = preprocess_text(search)
    code = vectorizer.transform([contents])
    
    # Calculate cosine similarity
    dist = cosine_similarity(code, keywords)
    
    # Get top 10 most similar articles
    top_indices = dist.argsort()[0, :-21:-1]
    result = data.iloc[top_indices]
    result1 = result[['Tanggal', 'Judul', 'Pengarang', 'Abstrak', 'URL']]
    
    return result1