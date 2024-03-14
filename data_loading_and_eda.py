import pandas as pd
from text_cleaning import clean_text
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('text.csv')
df = df.sample(n=1000, random_state=42)

df['text'] = df['text'].apply(clean_text)
df['text_length'] = df['text'].apply(len)

def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['text'].apply(calculate_sentiment)

def plot_sentiment_distribution(df):
    sns.histplot(df['sentiment'], kde=True)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()

plot_sentiment_distribution(df)

words = df['text'].str.split()
all_words = [word for words_list in words for word in words_list]
word_freq = Counter(all_words)

freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
freq_df = freq_df.sort_values(by='frequency', ascending=False)

print(freq_df.head(20))

plt.figure(figsize=(10, 6))
sns.barplot(data=freq_df.head(20), x='frequency', y='word')
plt.title('Top 20 Most Frequent Words')
plt.show()

df['sentiment'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)
df['sentiment_label'] = df['sentiment'].apply(lambda sentiment: 'positive' if sentiment > 0 else ('negative' if sentiment < 0 else 'neutral'))

print(df.head())

plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30)
plt.title('Sentiment Polarity Distribution')
plt.show()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text'),
        ('num', StandardScaler(), ['text_length'])
    ])

X = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment_label'], test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()