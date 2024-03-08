import pandas as pd
from text_cleaning import clean_text
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob

# Load the CSV file
df = pd.read_csv('text.csv')

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

# Print the first few rows of the DataFrame with the sentiment scores
print(df.head())


# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30)
plt.title('Sentiment Polarity Distribution')
plt.show()