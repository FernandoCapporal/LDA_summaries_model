import re
import html
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import gensim
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import random
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec


random.seed(42)
np.random.seed(42)


class RemoveLastParagraphTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return ['This award reflects'.join(text.split('This award reflects')[:-1]) if 'This award reflects' in text else text for text in X]


class RemoveHTMLTagsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_texts = []
        for text in X:
            text = html.unescape(text)
            text_without_html = BeautifulSoup(text, "html.parser").get_text()
            text_without_html = re.sub(r'\s+', ' ', text_without_html).strip()
            cleaned_texts.append(text_without_html)
        return cleaned_texts

        
class BertPreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stops = set(stopwords.words('english'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([' '.join([token.lower() for token in self.tokenizer.tokenize(text) if token.isalpha() and token.lower() not in self.stops]) for text in X]).reshape(-1, 1) 


class SimplePreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stops = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([' '.join([token.lower() for token in word_tokenize(text) if token.lower() not in self.stops and token.isalpha()]) for text in X]).reshape(-1, 1)


class TextPreprocessingSwitcher(BaseEstimator, TransformerMixin):
    def __init__(self, use_bert=False):
        self.use_bert = use_bert
        self.data = None
        self.simple_preprocess = SimplePreprocessTransformer()
        self.bert_preprocess = BertPreprocessTransformer()

    def fit(self, X, y=None):
        self.data = X
        return self

    def transform(self, X, y=None):
        if self.use_bert:
            return {'to_token': self.bert_preprocess.transform(X), 'summaries': self.data}
        else:
            return {'to_token': self.simple_preprocess.transform(X), 'summaries': self.data}


class LDAModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics=20, iterations=25):
        self.num_topics = num_topics
        self.iterations = iterations
        self.original_summaries = None
        self.summaries = None
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.top_topics = None
        self.coherence = None
        
    def fit(self, X, y=None):
        self.original_summaries = X['summaries']
        self.summaries = self.tokenization(X['to_token'].tolist())
        return self

    def transform(self, y=None):
        self.dictionary = Dictionary(self.summaries)
        self.dictionary.filter_extremes(no_below=50, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(summary) for summary in self.summaries]
        temp = self.dictionary[0]  
        id2word = self.dictionary.id2token
        self.model = LdaModel(corpus=self.corpus, id2word=id2word, 
                              iterations=self.iterations, num_topics=self.num_topics)
        self.top_topics = list(self.model.top_topics(self.corpus))
        coherence_model_lda = gensim.models.CoherenceModel(model=self.model, texts=self.summaries, 
                                                           dictionary=self.dictionary, coherence='c_v')
        self.coherence = coherence_model_lda.get_coherence()
        return self
    
    def tokenization(self, data):
        df = pd.DataFrame([[text] for text in data], columns=['text'])
        tokens = []
        for index, text in df.iterrows():
            tokens.append([])
            for word in text:
                tokens[index] = word[0].split()
        return tokens
    

class TopicLabelingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word2vec_model = None
        self.lda_model = None
        self.summaries = None

    def fit(self, X, y=None):
        self.lda_model = X.model
        self.summaries = X.summaries
        self.word2vec_model = Word2Vec(sentences=self.summaries, vector_size=100, window=5, min_count=1, workers=4)
        return self

    def transform(self, X=None):
        topics = [[(prob, word) for word, prob in word_probs] for _, word_probs in self.lda_model.show_topics(formatted=False)]
        topic_labels = self.label_topics(topics)
        return topic_labels

    def label_topics(self, topics):
        topic_labels = []

        for topic in topics:
            topic_words = [word for _, word in topic if word in self.word2vec_model.wv]

            if not topic_words:
                topic_labels.append(None)
                continue

            topic_vector = np.mean([self.word2vec_model.wv[word] for word in topic_words], axis=0)

            most_similar_word = None
            max_similarity = -1

            for word in self.word2vec_model.wv.key_to_index:
                similarity = 1 - cosine(topic_vector, self.word2vec_model.wv[word])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_word = word

            topic_labels.append(most_similar_word)

        return topic_labels