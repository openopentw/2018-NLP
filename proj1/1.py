import nltk
import numpy as np
import pandas as pd
from sklearn import *
import sys # argv
import xgboost as xgb

def read_train_test (path, cols):
    data = pd.read_json(path)
    x_data = data[cols].values
    y_data = data[['sentiment']].values
    return x_data, y_data

def read_sdfin (path, cols):
    data = pd.read_json(path)
    data = data[cols]
    return data

def read_glove (path):
    print('Reading GLOVE.')
    glove = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    print('Finish reading GLOVE.')
    return glove

def sentence_to_scores (sentence, dic):
    scores = []
    tokens = nltk.word_tokenize(sentence)
    for t in tokens:
        if t in dic:
            scores += [dic[t]]
    return scores

def preprocess_on_x (x_data, dic):
    X_data = []
    for x in x_data:
        scores = []
        if isinstance(x[0], str):
            scores = sentence_to_scores(x[0], dic)
        elif isinstance(x[0], list):
            for sentence in x[0]:
                scores += sentence_to_scores(sentence, dic)
        if len(scores) > 0:
            X_data += [[
                max(scores),
                min(scores),
                sum(scores) / len(scores)
            ]]
        else:
            X_data += [[0, 0, 0]]
    X_data = np.array(X_data)
    return X_data

if __name__ == '__main__':
    # Args
    # eg:
    #   NTUSD-Fin
    #       python .\1.py .\data\training_set.json .\data\test_set.json .\data\NTUSD-Fin\NTUSD_Fin_word_v1.0.json .\output\output.csv
    #   word2vec
    #       python .\1.py .\data\training_set.json .\data\test_set.json .\data\GoogleNews-vectors-negative300.bin .\output\output.csv
    #   GLOVE
    #       python .\1.py .\data\training_set.json .\data\test_set.json .\data\glove.6B\glove.6b.300d.txt .\output\output.csv
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    sdfin_path = sys.argv[3]
    # gg_w2v_path = sys.argv[3]
    # glove_path = sys.argv[3]
    output_path = sys.argv[4]

    x_train, y_train = read_train_test(train_path, [
        # 'sentiment'
        'snippet',
        # 'target',
        # 'tweet',
    ])
    x_test, y_test = read_train_test(test_path, [
        # 'sentiment'
        'snippet',
        # 'target',
        # 'tweet',
    ])

    sdfin = read_sdfin(sdfin_path, [
        # 'bear_cfidf',
        # 'bear_freq',
        # 'bull_cfidf',
        # 'bull_freq',
        # 'chi_squared',
        'market_sentiment',
        'token',
        # 'word_vec',
    ])
    sdfin_dict = sdfin.set_index('token')['market_sentiment'].to_dict()
    # glove = read_glove(glove_path)

    X_train = preprocess_on_x(x_train, sdfin_dict)
    X_test = preprocess_on_x(x_test, sdfin_dict)

    model = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3)
    model.fit(X_train, y_train[:,0])

    pred = model.predict(X_test)
    error = metrics.mean_squared_error(pred, y_test[:,0])
    print('MSE: {}'.format(error)) # MSE: 0.10734628588677432
    # all-zero: MSE: 0.15586947160883283
    np.savetxt(output_path, pred, delimiter=',')
