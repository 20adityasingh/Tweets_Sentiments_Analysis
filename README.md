Twitter Sentiment Analysis Project
This project uses the Sentiment140 dataset, which contains 1.6 million tweets, to perform sentiment analysis. The primary objective is to build and evaluate a variety of neural network models for classifying tweets as either positive or negative.

Required Libraries
This project relies on several key Python libraries for data manipulation, visualization, and deep learning.

# Core Libraries
import pandas as pd
import numpy as np
import re

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Scikit-learn for ML utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TensorFlow/Keras for Neural Networks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, SimpleRNN, LSTM, GRU, Bidirectional, GlobalMaxPool1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# NLTK and SpaCy for NLP tasks
import spacy
import nltk
from nltk.corpus import stopwords

Data Preparation
The workflow begins with data loading and extensive preprocessing to prepare the text for modeling.

Loading Data: The training.1600000.processed.noemoticon.csv dataset is loaded.

Cleaning Text: A custom function cleans each tweet by converting it to lowercase, removing URLs, user mentions, and hashtags, and then lemmatizing the text while filtering out stopwords.

Feature Engineering: New features, such as character and word counts, are created from the cleaned tweets to provide additional information to the models.

Exploratory Data Analysis
Before modeling, the data is visualized to understand its characteristics:

Sentiment Distribution: A pie chart is used to show the balance between positive and negative tweets.

Text Length Analysis: Distribution plots help visualize the differences in word and character counts for each sentiment class.

Modeling and Evaluation
A wide range of neural network architectures were developed to find the most effective model for this task. The data was split into training, validation, and test sets for robust evaluation.

Vectorization-Based Models
These models use traditional text vectorization methods as input to a simple Dense neural network.

CountVectorizer (Binary): Represents text based on the presence of words.

CountVectorizer (Frequency): Represents text based on word counts.

TF-IDF Vectorizer: Represents text based on term frequency-inverse document frequency.

Sequence-Based Models
These models use word embeddings to capture semantic relationships between words.

Index-Based Encoding: Converts text into sequences of integers, which are fed into a Dense network.

Word Embedding with a Dense Network: Uses an Embedding layer to learn word representations before passing them to a Dense network.

Recurrent Neural Networks (RNNs): More complex models that process the sequence of words.

SimpleRNN: A basic recurrent network.

LSTM (Long Short-Term Memory): A more advanced RNN capable of learning long-range dependencies.

GRU (Gated Recurrent Unit): A variation of LSTM that is computationally more efficient.

Bidirectional Models (LSTM & GRU): These models process the text sequence in both forward and backward directions, capturing more context. A particularly complex Bidirectional LSTM model was also tested.

Each model was compiled with the Adam optimizer and binary_crossentropy loss function and was evaluated based on its accuracy.
