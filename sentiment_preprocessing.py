# End to end script for training a sentiment classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import io
import pickle


def load_data(filepath):
    '''takes a .csv filename and loads it'''
    return pd.read_csv(filepath)


def split_df(df, keep_cols, reason_cols, drop_cols):
    '''splits a dataframe into 3 by the 3 different column lists passed'''
    df = df.drop_duplicates(subset=keep_cols)
    keep_df = df[keep_cols]
    reason_df = df[reason_cols]
    drop_df = df[drop_cols]
    return keep_df, reason_df, drop_df


def create_labels(df, label_col):
    '''pops the label_col from the given df and returns it'''
    return df.pop(label_col)


def infrequent_words(word_df, threshhold):
    '''takes a dataframe of words and counts, and returns the dataframe with
    all words appearing less than the threshhold removed'''
    return list(word_df.loc[word_df['count'] <= threshhold, 'word'])


def noise_words(word_df, threshhold):
    '''takes a dataframe of words and ratios, and returns the dataframe with
    all words with ratio below the threshhold removed'''
    return list(word_df.loc[abs(word_df['ratio']) <= threshhold, 'word'])


def dummy_tokenizer(doc):
    '''dummy tokenizer for tfidf'''
    return doc


def encode_labels(labels):
    '''transforms labels to 1 for positive and 0 for negative'''
    labels.loc[labels == 'positive'] = 1
    labels.loc[labels == 'negative'] = 0
    return labels.astype('int')


def write_labels(filename, labels):
    '''writes labels to disk at given filename'''
    pickle.dump(labels, open(filename, 'wb'))


def write_matrix(filename, mtx):
    '''writes a doc term matrix at a given filename'''
    io.mmwrite(filename, mtx)


class TextProcessor(BaseEstimator, TransformerMixin):
    '''This class performs a variety of preprocessing steps on an array of texts
    including converting to lowercase, removing punctuation and contractions,
    tokenizing the text, removing stopwords, and lemmatizing the text'''
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, corpus):
        '''takes a list of texts and transforms it with the steps listed
        above'''
        corpus = corpus.str.lower()
        corpus = corpus.map(lambda x: ''.join(ch for ch in x if ch not in
                            punctuation))
        corpus = corpus.map(lambda x: contractions.fix(x))
        corpus = corpus.map(word_tokenize)
        corpus = corpus.map(lambda x: [w for w in x if w not in
                            stopwords.words('english')])
        lem = WordNetLemmatizer()
        corpus = corpus.map(lambda x: [lem.lemmatize(w, pos='n') for w in x])
        corpus = corpus.map(lambda x: [lem.lemmatize(w, pos='v') for w in x])
        corpus = corpus.map(lambda x: [lem.lemmatize(w, pos='a') for w in x])
        return corpus


class RemoveNoise(BaseEstimator, TransformerMixin):
    '''This class removes words that are very rare or offer little for
    classification'''
    def __init__(self):
        pass

    def calculate_word_df(self, X, y):
        '''creates a word dataframe with a word column, a counts column, and a
        ratio column from the given features and labels'''
        tokenlist = []
        pos_tokens = X.loc[y == 'positive']
        neg_tokens = X.loc[y == 'negative']
        for token_list in X['text']:
            tokenlist.extend(token_list)
        all_tokens = list(set(tokenlist))

        pos_percent = {}
        neg_percent = {}
        token_count = []
        ratio = []
        for token in all_tokens:
            token_count.append(sum(X['text'].map(lambda x: token in x)))
            pos_percent[token] = np.mean(pos_tokens['text'].map(lambda x:
                                                                token in x))
            neg_percent[token] = np.mean(neg_tokens['text'].map(lambda x:
                                                                token in x))
            ratio.append(pos_percent[token] - neg_percent[token])

        word_df = pd.DataFrame({'word': all_tokens, 'count': token_count,
                                'ratio': ratio})
        return word_df

    def fit(self, X, y, frequency_thresh, noise_thresh):
        '''calculates the word dataframe on train and test data, and creates a
         list of words below the frequency threshhold and list of words with
         ratio below the noise threshhold'''
        word_df = self.calculate_word_df(X, y)
        infreq_list = infrequent_words(word_df, frequency_thresh)
        noise_list = noise_words(word_df, noise_thresh)
        self.infreq_list = infreq_list
        self.noise_list = noise_list
        return self

    def transform(self, X, y=None):
        '''removes words in infrequent and noise lists'''
        X['text'] = X['text'].map(lambda x: [w for w in x if w not in
                                  self.infreq_list])
        X['text'] = X['text'].map(lambda x: [w for w in x if w not in
                                  self.noise_list])
        return X


if __name__ == "__main__":

    df = load_data('data/twitter-airline-sentiment/Tweets.csv')
    df = df.loc[df['airline_sentiment'] != 'neutral', ]
    train, test = train_test_split(df, test_size=.2, random_state=42)

    # This can be altered to balance the training set to have equal # of
    # instances for the positive and negative class. This is NOT
    # recommended and tends to lead to overfitting.

    balance_dataset = False

    if balance_dataset:
        pos = train.loc[df['airline_sentiment'] == 'positive']
        neg = train.loc[df['airline_sentiment'] == 'negative']
        new_pos = pos.sample(neg.shape[0], replace=True, axis=0)
        train = neg.append(new_pos).sample(frac=1)

    keep_cols = ['tweet_id', 'airline_sentiment',
                 'airline_sentiment_confidence', 'retweet_count', 'text']
    reason_cols = ['negativereason', 'negativereason_confidence']
    drop_cols = ['airline', 'airline_sentiment_gold', 'name',
                 'negativereason_gold', 'tweet_coord', 'tweet_created',
                 'tweet_location', 'user_timezone']

    # Split dataframes into groups based on use case
    train_keep, train_reason, train_drop = split_df(train, keep_cols,
                                                    reason_cols, drop_cols)
    test_keep, test_reason, test_drop = split_df(test, keep_cols,
                                                 reason_cols, drop_cols)

    # Create target arrays, remove targets from features
    train_labels = create_labels(train_keep, 'airline_sentiment')
    test_labels = create_labels(test_keep, 'airline_sentiment')

    # Save these dataframes for later use in RNN model
    train_keep.to_csv('data/train_keep.csv')
    test_keep.to_csv('data/test_keep.csv')

    # preprocess train and test sets by removing punc, stopwords, lemmatizing
    processor = TextProcessor()
    train_preprocessed = train_keep.copy()
    test_preprocessed = test_keep.copy()
    train_tokens = processor.transform(train_keep['text'])
    train_preprocessed['text'] = train_tokens
    test_tokens = processor.transform(test_keep['text'])
    test_preprocessed['text'] = test_tokens

    # Save dataframes for later use in RNN model
    train_preprocessed.to_csv('data/train_preprocessed.csv')
    test_preprocessed.to_csv('data/test_preprocessed.csv')

    # removes infrequent words and words with low predictive power
    train_tokens = train_preprocessed.copy()
    test_tokens = test_preprocessed.copy()
    noise_remover = RemoveNoise()
    # This removes words that occur less than 5 times and a ratio below .002
    noise_remover.fit(train_tokens, train_labels, 5, 0.002)
    noise_remover.transform(train_tokens)
    noise_remover.transform(test_tokens)

    # Save dataframes for later use in RNN model
    train_tokens.to_csv('data/train_tokens.csv')
    test_tokens.to_csv('data/test_tokens.csv')

    # Save visualizations
    word_df = noise_remover.calculate_word_df(train_preprocessed, train_labels)
    word_df.sort_values('ratio', inplace=True, ascending=False)
    fig = plt.figure()
    sns.distplot(word_df['ratio'])

    # Save plot and wordratio dataframe
    fig.savefig('figures/ratio_distribution.png')
    word_df.to_csv('data/word_df.csv')

    # Check if we have created any empty lists
    print(train_labels.loc[
          train_keep['text'].astype(str) == '[]'].value_counts())

    # Create TFIDF doc-word matrix
    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy_tokenizer,
                            preprocessor=dummy_tokenizer, token_pattern=None)
    train_doc_word = tfidf.fit_transform(train_tokens['text'])
    test_doc_word = tfidf.transform(test_tokens['text'])

    # One hot encode targets
    train_labels = encode_labels(train_labels)
    test_labels = encode_labels(test_labels)

    # Write matrices and labels to disk
    write_matrix('data/train_matrix.mtx', train_doc_word)
    write_matrix('data/test_matrix.mtx', test_doc_word)
    write_labels('data/train_labels.p', train_labels)
    write_labels('data/test_labels.p', test_labels)
