import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download the stopwords from NLTK
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and stem the words
    text = ' '.join(ps.stem(word) for word in text.split() if word not in stop_words)
    return text