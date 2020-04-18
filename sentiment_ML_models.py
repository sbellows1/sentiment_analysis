import pandas as pd
from scipy import io
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from itertools import count
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.metrics import f1_score
from keras.models import load_model
from string import punctuation
import contractions
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os


def load_matrix(filename):
    '''loads a .mtx file from filename'''
    return io.mmread(filename)


def load_labels(filename):
    '''loads a .p file from filename'''
    return pickle.load(open(filename, 'rb'))


def train_clf_model(model, doc_word_matrix, labels, scoring='accuracy',
                    cv=5, verbose=5, return_estimator=True):
    '''Trains a classifier model. model is the classifier to be trained,
    doc_word_matrix is a document term matrix, labels is an array of targets,
    scoring is the scoring function, cv is the number of cv folds,
    verbose is the level of verbosity, and return_estimator determines whether
    the estimator is returned. (This should be set to True)'''
    return cross_validate(model, doc_word_matrix.toarray(), labels,
                          scoring=scoring, cv=cv, verbose=verbose,
                          return_estimator=return_estimator)


def load_data(filepath):
    '''loads a csv file from filepath'''
    return pd.read_csv(filepath)


def word_to_id(token_list):
    '''takes a set of words and assigns them each to a unique integer id'''
    id_dict = defaultdict((count().__next__))
    for token in token_list:
        id_dict[token] = id_dict[token]
    return id_dict


def get_stopword_tokens(df):
    '''takes a dataframe with a 'text' column. prepares text for modeling by
    converting text to lowercase, removing punctuation, removing contractions,
    and tokenizing the text.'''
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].map(lambda x: ''.join(ch for ch in x if
                                ch not in punctuation))
    df['text'] = df['text'].map(lambda x: contractions.fix(x))
    df['text'] = df['text'].map(word_tokenize)
    return df


def tokens_to_sequence(tokens):
    '''takes a dataframe with a 'text' column, where each field in the 'text'
    column is a list. transforms these lists of tokens to lists of integers
    based on each token's unique id'''
    tokenlist = []
    for token in tokens['text']:
        tokenlist.extend(token)
    id_dict = word_to_id(list(set(tokenlist)))
    sequence_list = tokens['text'].map(lambda x: [id_dict[w] + 1 for w in x])
    return sequence_list, id_dict


def test_token_update(tokens, id_dict):
    '''takes a dataframe with a 'text' column of lists, and a word to id
    dictionary. Transforms the lists of tokens to integers using the passed in
    dictionary'''
    sequence_list = tokens['text'].map(lambda x: [id_dict[w] + 1 if w in
                                       id_dict.keys() else 0 for w in x])
    return sequence_list


def pad_test_sequence(sequence, train_sequence):
    '''takes an array of sequences of test data and a padded array of sequences
    of train data, pads the test data to match the train data.'''
    return pad_sequences(sequence, len(max(train_sequence, key=len)),
                         'int', 'post', 'post', 0)


def pad_sequence(sequence):
    '''takes an array of sequences of train data and pads them to equal
    length'''
    return pad_sequences(sequence, len(max(sequence, key=len)),
                         'int', 'post', 'post', 0)


def build_train_model(sequence, y, id_dict, embedding_size,
                      lstm_size, dropout, epochs, batch_size, filename,
                      figname):
    '''trains an RNN. Takes an array of sequences of equal length as training
    data, an array of labels as targets, a dictionary mapping words to
    integers, the size of the embedding and LSTM layers, the percent of nodes
    to dropout, the number of epochs, the batch_size, the filename to save the
    model to, and the filename to save the figure to'''

    model = Sequential()
    model.add(Embedding(max(id_dict.values()) + 2, embedding_size,
              mask_zero=True, input_length=len(max(sequence, key=len))))
    model.add(LSTM(lstm_size, activation='tanh',
              recurrent_activation='sigmoid', dropout=dropout, use_bias=True,
              unit_forget_bias=True))
    model.add(Dense(1, activation='sigmoid', use_bias=True))
    model.add(Dropout(0.2))

    adam = optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=10)
    mc = ModelCheckpoint(filename, monitor='val_accuracy',
                         save_best_only=True)

    history = model.fit(x=sequence, y=y, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_split=.2,
                        callbacks=[es, mc])

    fig = plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    fig.savefig(figname)


def test_metric_clf(clf, test, test_labels):
    '''takes a classifier, test data, and test labels, and returns the f1 Score
    for said classifier on the test data'''
    return f1_score(test_labels,
                    clf['estimator'][0].predict(test.toarray()))


def test_metric_rnn(rnn, test, test_labels):
    '''takes an rnn, test data, and test labels, and returns the f1 score for
    said rnn on the test data'''
    return f1_score(test_labels,
                    pd.Series(rnn.predict(test).flatten()).map(round))


def save_results(df, filename, figname):
    '''takes a dataframe of models and test scores, a filename to save the
    results to, and a filename to save the figure to.'''
    df.to_csv(filename)
    fig = plt.figure()
    df.plot().bar(x='model', y='F1 Score')
    fig.savefig(figname)


if __name__ == "__main__":

    if not(os.path.exists('figures')):
        os.mkdir('figures')
    if not(os.path.exists('models')):
        os.mkdir('models')
    train_matrix = load_matrix('data/train_matrix.mtx')
    test_matrix = load_matrix('data/test_matrix.mtx')
    train_labels = load_labels('data/train_labels.p')
    test_labels = load_labels('data/test_labels.p')

    metric = 'f1'

    multi_clf = MultinomialNB()
    gauss_clf = GaussianNB(priors=[.79, .21])
    gb = GradientBoostingClassifier(max_depth=5)
    model_list = [multi_clf, gauss_clf, gb]
    trained_models = [train_clf_model(m, train_matrix, train_labels,
                                      scoring=metric) for m in model_list]

    model_df = pd.DataFrame({'model': ['MultinomialNB', 'GaussianNB',
                                       'GradientBoosting'],
                             'metric': [m['test_score'][0]
                                        for m in trained_models]})
    model_df.to_csv('models/ML_models.csv')

    train_keep = load_data('data/train_keep.csv')
    test_keep = load_data('data/test_keep.csv')
    train_preprocessed = load_data('data/train_preprocessed.csv')
    test_preprocessed = load_data('data/test_preprocessed.csv')
    test_tokens = load_data('data/test_tokens.csv')
    train_tokens = load_data('data/train_tokens.csv')

    train_stopwords = get_stopword_tokens(train_keep)
    test_stopwords = get_stopword_tokens(test_keep)

    sequence_tokens, id_dict_tokens = tokens_to_sequence(train_tokens)
    sequence_pre, id_dict_pre = tokens_to_sequence(train_preprocessed)
    sequence_stop, id_dict_stop = tokens_to_sequence(train_stopwords)

    sequence_tokens = pad_sequence(sequence_tokens)
    sequence_pre = pad_sequence(sequence_pre)
    sequence_stop = pad_sequence(sequence_stop)

    test_token_sequence = test_token_update(test_tokens, id_dict_tokens)
    test_pre_sequence = test_token_update(test_preprocessed, id_dict_pre)
    test_stop_sequence = test_token_update(test_stopwords, id_dict_stop)

    test_token_sequence = pad_test_sequence(test_token_sequence,
                                            sequence_tokens)
    test_pre_sequence = pad_test_sequence(test_pre_sequence, sequence_pre)
    test_stop_sequence = pad_test_sequence(test_stop_sequence, sequence_stop)

    y = train_labels
    embedding_size = 100
    LSTM_size = 128
    dropout = .2
    epochs = 5
    batch_size = 100

    # Adjust model architecture to different models
    build_train_model(sequence_tokens, y, id_dict_tokens,
                      embedding_size, LSTM_size, dropout, epochs, batch_size,
                      'models/token.h5', 'figures/token.png')
    build_train_model(sequence_pre, y, id_dict_pre,
                      embedding_size, LSTM_size, dropout, epochs, batch_size,
                      'models/preprocess.h5', 'figures/preprocess.png')
    build_train_model(sequence_stop, y, id_dict_stop,
                      embedding_size, LSTM_size, dropout, epochs, batch_size,
                      'models/stopwords.h5', 'figures/stopwords.png')

    token_model = load_model('models/token.h5')
    preprocess_model = load_model('models/preprocess.h5')
    stopword_model = load_model('models/stopwords.h5')

    rnn_list = [token_model, preprocess_model, stopword_model]
    rnn_tests = [test_token_sequence, test_pre_sequence, test_stop_sequence]

    # Calculate F1 for both model types
    clf_metric = [test_metric_clf(m, test_matrix, test_labels)
                  for m in trained_models]
    rnn_metric = [test_metric_rnn(m, t, test_labels) for m, t in
                  zip(rnn_list, rnn_tests)]
    results_df = pd.DataFrame({'model': ['MultinomialNB', 'GaussianNB',
                                         'GradientBoosting']+[
        'Token model', 'Preprocessed model', 'Model with stopwords'],
        'F1 Score': clf_metric + rnn_metric})
    save_results(results_df, 'models/test_f1.csv', 'figures/test_f1_plot.png')
