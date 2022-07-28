import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from abc import ABC, abstractmethod

class utils(ABC):
    @abstractmethod
    def remove_punctuation(row):
        row.overview = ''.join(ch for ch in row.overview if ch not in exclude)
        return row

    @abstractmethod
    def lemmatize_text(row):
        lemmatizer = WordNetLemmatizer()
        row.overview = " ".join(lemmatizer.lemmatize(word) for word in    row.overview.split())
        return row
