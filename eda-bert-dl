#!/usr/bin/env python
# coding: utf-8








# #eda





import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
from collections import Counter
import plotly.figure_factory as ff

# Load the dataset
df = pd.read_csv(r'C:\Users\Krishnna Nikhil\Downloads\ML Assignment Dataset - Train.csv')

# Basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Check for missing values and drop rows with missing "tweet_text"
df = df.dropna(subset=['tweet_text'])

# Fill missing values
df['emotion_in_tweet_is_directed_at'].fillna('No emotion', inplace=True)
df['is_there_an_emotion_directed_at_a_brand_or_product'].fillna('No emotion toward brand or product', inplace=True)

# Create 'tweet_text_length' column
df['tweet_text_length'] = df['tweet_text'].apply(len)

# Summary statistics for numeric columns
numerical_summary = df.describe()
print("\nSummary Statistics for Numeric Columns:")
print(numerical_summary)

# Check for missing values again
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Class distribution
class_distribution = df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Distribution of emotions
emotion_distribution = df['emotion_in_tweet_is_directed_at'].value_counts()
print("\nDistribution of Emotions:")
print(emotion_distribution)

# Text length statistics
text_length_stats = df.groupby('is_there_an_emotion_directed_at_a_brand_or_product')['tweet_text_length'].describe()
print("\nText Length Statistics:")
print(text_length_stats)

# Correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='is_there_an_emotion_directed_at_a_brand_or_product', data=df, palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Emotion Directed at a Brand or Product')
plt.ylabel('Count')
plt.show()

# Visualize the distribution of emotions and products
plt.figure(figsize=(12, 8))

# Distribution of emotions
plt.subplot(2, 1, 1)
sns.countplot(x='emotion_in_tweet_is_directed_at', data=df, palette='pastel')
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')

# Distribution of products
plt.subplot(2, 1, 2)
sns.countplot(x='emotion_in_tweet_is_directed_at', hue='is_there_an_emotion_directed_at_a_brand_or_product', data=df, palette='muted')
plt.title('Distribution of Products by Emotion')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Boxplot to check for outliers in text length
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_there_an_emotion_directed_at_a_brand_or_product', y='tweet_text_length', data=df, palette='Set3')
plt.title('Text Length by Emotion')
plt.xlabel('Emotion Directed at a Brand or Product')
plt.ylabel('Text Length')
plt.show()

# Text length distribution by emotion and product
plt.figure(figsize=(12, 6))
sns.histplot(df, x='tweet_text_length', hue='is_there_an_emotion_directed_at_a_brand_or_product', multiple="stack", binwidth=50, palette='coolwarm')
plt.title('Text Length Distribution by Emotion and Product')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

# Distribution of Number Of Words in tweet_text
hist_data = [df['tweet_text'].apply(lambda x: len(str(x).split()))]
group_labels = ['Num_words_tweet_text']
fig = ff.create_distplot(hist_data, group_labels, show_curve=False)
fig.update_layout(title_text='Distribution of Number Of Words in tweet_text')
fig.update_layout(autosize=False, width=900, height=700, paper_bgcolor="LightSteelBlue")
fig.show()

# Kernel Distribution of Number Of Words in Tweet Text
plt.figure(figsize=(12, 6))
sns.kdeplot(df['tweet_text'].apply(lambda x: len(str(x).split())), shade=True, color="r").set_title('Kernel Distribution of Number Of Words in Tweet Text')
sns.kdeplot(df['emotion_in_tweet_is_directed_at'].apply(lambda x: len(str(x).split())), shade=True, color="b")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply clean_text function
df['tweet_text'] = df['tweet_text'].apply(lambda x: clean_text(x))
df['emotion_in_tweet_is_directed_at'] = df['emotion_in_tweet_is_directed_at'].apply(lambda x: clean_text(x))

# Create 'temp_list' containing lists of words in 'tweet_text'
df['temp_list'] = df['tweet_text'].apply(lambda x: str(x).split())

# Count the most common words in 'tweet_text'
top_words_tweet_text = Counter([item for sublist in df['temp_list'] for item in sublist])
common_words_tweet_text = pd.DataFrame(top_words_tweet_text.most_common(20), columns=['Common_words', 'count'])

# Visualization of common words in 'tweet_text'
fig_tweet_text = px.bar(common_words_tweet_text, x="count", y="Common_words", title='Common Words in Tweet Text', orientation='h', width=700, height=700, color='Common_words')
fig_tweet_text.show()

# Remove stopwords from 'temp_list'
stopwords_set = set(stopwords.words('english'))
df['temp_list'] = df['temp_list'].apply(lambda x: [word for word in x if word not in stopwords_set])

# Count the most common words in 'tweet_text' after removing stopwords
top_words_no_stopwords = Counter([item for sublist in df['temp_list'] for item in sublist])
common_words_no_stopwords = pd.DataFrame(top_words_no_stopwords.most_common(20), columns=['Common_words', 'count'])
common_words_no_stopwords = common_words_no_stopwords.iloc[1:, :]

# Visualization of common words in 'tweet_text' after removing stopwords
fig_no_stopwords = px.treemap(common_words_no_stopwords, path=['Common_words'], values='count', title='Tree of Most Common Words in Tweet Text (No Stopwords)')
fig_no_stopwords.show()

# Create 'temp_list1' containing lists of words in 'emotion_in_tweet_is_directed_at'
df['temp_list1'] = df['emotion_in_tweet_is_directed_at'].apply(lambda x: str(x).split())

# Remove stopwords from 'temp_list1'
df['temp_list1'] = df['temp_list1'].apply(lambda x: [word for word in x if word not in stopwords_set])
















# #bert model



import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Krishnna Nikhil\Downloads\ML Assignment Dataset - Train.csv')
    df = df.rename(columns={"tweet_text": "text", "emotion_in_tweet_is_directed_at": "emotion", "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment"})
    df = df.dropna(subset=['emotion'])
    train = df[['text', 'sentiment']].copy()
    train['text'].fillna('No text', inplace=True)

    def cleantext(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub("\'", "", text)
        return text

    train['text'] = train['text'].apply(cleantext)

    train["text"] = train["text"].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))
    train["text"] = train["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words("english")]))

    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    label_mapping = {'Positive emotion': 0, 'Negative emotion': 1, 'No emotion toward brand or product': 2}

    train_df['label'] = train_df['sentiment'].map(label_mapping)
    val_df['label'] = val_df['sentiment'].map(label_mapping)

    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])

    max_len = 128

    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

            encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_len,
                                       padding='max_length', truncation=True)

            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': label
            }

    train_dataset = CustomDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_df['text'].values, val_df['label'].values, tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy}')

    report = classification_report(all_labels, all_preds, zero_division=1)
    print(report)















# #deep learning model




import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import gensim
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Krishnna Nikhil\Downloads\ML Assignment Dataset - Train.csv')
df = df.rename(columns={"tweet_text": "text", "emotion_in_tweet_is_directed_at": "emotion", "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment"})
df = df.dropna(subset=['emotion'])
train = df[['text', 'sentiment']].copy()
train['text'].fillna('No text', inplace=True)

def cleantext(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\'", "", text)
    return text

train['text'] = train['text'].apply(cleantext)
train["text"] = train["text"].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
train["text"] = train["text"].apply(lambda wrd: ''.join(wrd))
train["text"] = train["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words("english")]))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

temp = []
data_to_list = train['text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(data_to_list[i])
data_wordslem = list(sent_to_words(temp))

lemmatizer = nltk.stem.WordNetLemmatizer()

for i in range(len(data_wordslem)):
    for j in range(len(data_wordslem[i])):
        data_wordslem[i][j] = lemmatizer.lemmatize(data_wordslem[i][j], pos="v")

data = []
for i in range(len(data_wordslem)):
    data.append(detokenize(data_wordslem[i]))

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)

embedding_layer = Embedding(input_dim=max_words, output_dim=64)

model2 = Sequential()
model2.add(embedding_layer)
model2.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
model2.add(layers.Dense(3, activation='softmax'))

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train['sentiment'] = train['sentiment'].str.lower()

invalid_labels = train['sentiment'][~train['sentiment'].isin(label_mapping.keys())].unique()
train['sentiment'].replace(invalid_labels, 'no emotion toward brand or product', inplace=True)
train['sentiment'] = train['sentiment'].map(label_mapping)

print(train['sentiment'].unique())
print(model2.summary())

try:
    history = model2.fit(
        x=tweets,
        y=train['sentiment'],
        epochs=70,
        validation_split=0.2,
        shuffle=True
    )

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

except Exception as e:
    print("Error during training:", e)
