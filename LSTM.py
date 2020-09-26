import torch
from torch import nn
from torchtext import data
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import urllib.request
import zipfile
import spacy
from torch import optim

drive.mount('/content/drive')

path = "/content/drive/My Drive/Colab Notebooks/Fake Tweets/training.1600000.processed.noemoticon.csv"
data = pd.read_csv(path, engine='python',header=None)

# Add 2 categories, because original categories are 0 and 4. Convert them to 0,1
data["sentiment_cat"] = data[0].astype('category')
data["sentiment"] = data["sentiment_cat"].cat.codes
data.to_csv("/content/drive/My Drive/Colab Notebooks/Fake Tweets/training-processed.csv", header=None, index=None)
data.sample(10000).to_csv("/content/drive/My Drive/Colab Notebooks/Fake Tweets/train-processed-sample.csv", header=None, index=None)

path = "/content/drive/My Drive/Colab Notebooks/Fake Tweets/training-processed-sample.csv"
data_tweets = pd.read_csv(path, engine="python", header=None)

# Create Fields using torchtext
LABEL = data.LabelField()
TWEET = data.Field(tokenize = 'spacy', lower=True)

# Match fields with the csv
fields = [('score', None),
          ('id', None), 
          ('date', None), 
          ('query', None), 
          ('name', None),
          ('tweet', TWEET), 
          ('category', None),
          ('label', LABEL)]

# Create the dataset
twitterDataset = data.TabularDataset(
    path= "/content/drive/My Drive/Colab Notebooks/Fake Tweets/train-processed-sample.csv",
    format="CSV",
    fields = fields,
    skip_header = False
)
# Split into train, validation and test set
train, valid, test = twitterDataset.split(split_ratio=[0.8, 0.1, 0.1],stratified=True, strata_field='label')

len(train), len(valid), len(test)

#Print contents of the dictionary  containing tweet number 16(random)
vars(train.examples[16])

# Build vocabulary
# Set the max size of the vocab to not occupy all RAM
vocab_size = 20000
TWEET.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)
len(TWEET.vocab)

# Print the most common words found
TWEET.vocab.freqs.most_common(10)
# Set device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create iteratora to provi with batches
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test),
    batch_size = 32,
    device = device,
    sort_key = lambda x: len(x.tweet),
    sort_within_batch = False)
# Create model with 3 layers
class FirstLSTM(nn.Module):
  def __init__(self, hidden_size, embedding_dim, vocab_size):
    super(FirstLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.encoder = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_size,num_layers=1)
    self.predictor = nn.Linear(hidden_size, 2)

  def forward(self, seq):
    output, (hidden, _) = self.encoder(self.embedding(seq))
    preds = self.predictor(hidden.squeeze(0))
    return preds

model = FirstLSTM(100, 300, 20002)
model.to(device)
# TRAIN
optimizer = optim.Adam(model.parameters(), lr = 2e-2)
criterion = nn.CrossEntropyLoss()
train_loss_hist = []
valid_loss_hist = []
def train(epochs, model, optimizer, criterion, train_iterator, valid_iterator):
  for epoch in range(1, epochs+1):

    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(train_iterator):
      optimizer.zero_grad()
      predict = model(batch.tweet)
      loss = criterion(predict, batch.label)
      loss.backward()
      optimizer.step()
      train_loss += loss.data.item() * batch.tweet.size(0)
    train_loss /= len(train_iterator)

    # Validate each epoch because its very fast
    model.eval()
    with torch.no_grad():
      for batch_idx, batch in enumerate(valid_iterator):
        predict = model(batch.tweet)
        loss = criterion(predict, batch.label)
        valid_loss += loss.data.item() * batch.tweet.size(0)
      valid_loss /= len(valid_iterator)

    train_loss_hist.append(train_loss)
    valid_loss_hist.append(valid_loss)
    print(f"Epoch: {epoch}, Training Loss: {train_loss:.4f},Validation Loss: {valid_loss:.4f}")
  print("Training Finished.")
  return train_loss_hist, valid_loss_hist

train_loss_hist, valid_loss_hist = train(100, model, optimizer, criterion, train_iterator, valid_iterator)

# Plot train loss and validation loss
import numpy as np
num_eponhs = np.arange(len(train_loss_hist))
plt.figure(figsize=(10,10))
plt.plot(num_eponhs, train_loss_hist, label='train_loss')
plt.plot(num_eponhs, valid_loss_hist, label='val_loss')
plt.legend()

# Inference performed like that
def classify_tweet(tweet):
 categories = {0: "Negative", 1:"Positive"}
 processed = TWEET.process([TWEET.preprocess(tweet)])
 processed = processed.to("cuda")
 return categories[model(processed).argmax().item()]

# Example of classification
pred = classify_tweet("I hate you")