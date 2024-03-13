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

# Load the full CSV file
df = pd.read_csv('text.csv')

# Create a subset of the data
df = df.sample(n=1000, random_state=42)  # Change n to the number of samples you want

# Apply the clean_text function to the 'text' column
df['text'] = df['text'].apply(clean_text)

# Split the cleaned text into words
words = df['text'].str.split()

# Flatten the list of words in the cleaned text
all_words = [word for words_list in words for word in words_list]

# Count the frequency of each word
word_freq = Counter(all_words)

# Create a DataFrame from the dictionary
freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])

# Sort the DataFrame by frequency
freq_df = freq_df.sort_values(by='frequency', ascending=False)

# Display the top 20 most frequent words
print(freq_df.head(20))

# Plot the top 20 most frequent words
plt.figure(figsize=(10, 6))
sns.barplot(data=freq_df.head(20), x='frequency', y='word')
plt.title('Top 20 Most Frequent Words')
plt.show()

# Calculate sentiment polarity of the cleaned text
df['sentiment'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Create sentiment labels based on sentiment polarity
df['sentiment_label'] = df['sentiment'].apply(lambda sentiment: 'positive' if sentiment > 0 else ('negative' if sentiment < 0 else 'neutral'))

# Print the first few rows of the DataFrame with the sentiment scores
print(df.head())

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30)
plt.title('Sentiment Polarity Distribution')
plt.show()

# Initialize a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the cleaned text data and transform it into TF-IDF vectors
tfidf_vectors = vectorizer.fit_transform(df['text'])

# Split your data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, df['sentiment_label'], test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Initialize a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Evaluate the model with the best parameters on the test set
y_pred = grid_search.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()