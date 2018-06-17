import nltk
import numpy as np
import pandas as pd
import sys # argv
import xgboost as xgb
import gensim

from sklearn import metrics
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    print('Finish reading GLOVE.')
    return glove

def sentence_to_embds (sentence, dic):
    embds = []
    tokens = nltk.word_tokenize(sentence)
    for t in tokens:
        if t in dic:
            embds += [dic[t]]
    return np.array(embds)

def make_embds (x_data, dic):
    X_data = np.zeros((1, 900))
    for x in x_data:
        embds = np.zeros((1, 300))
        if isinstance(x[0], str):
            embds = sentence_to_embds(x[0], dic)
        elif isinstance(x[0], list):
            for sentence in x[0]:
                embd = sentence_to_embds(sentence, dic)
                if embd.size > 0:
                    embds = np.append(embds, embd, axis=0)
            embds = embds[1:]

        # print(embds.shape)

        if embds.size > 0:
            new_embd = np.append(embds.max(axis=0), embds.min(axis=0))
            new_embd = np.append(new_embd, embds.mean(axis=0))
            new_embd = new_embd.reshape((1, 900))
            X_data = np.append(X_data, new_embd, axis=0)
        else:
            X_data = np.append(X_data, np.zeros((1, 900)), axis=0)
    # print(X_data[1:].shape)
    return X_data[1:]

def make_tag_index (tags, thres=1):
    tags_cnt = Counter(tags)
    tags_thres = [t if tags_cnt[t] > thres else '$OTHER' for t in tags]
    tags_thres_uni = list(set(tags_thres))
    tag_index = {t : tags_thres_uni.index(t) for t in tags_thres_uni}
    return tag_index

def make_tag_onehot (tags, tag_index):
    onehot = np.zeros((len(tags), len(tag_index)))
    for i,t in enumerate(tags):
        if t in tag_index:
            onehot[i][tag_index[t]] = 1
        else:
            onehot[i][tag_index['$OTHER']] = 1
    return onehot

def preprocess_on_x (x_data, dic, tag_index):
    word_embd = make_embds(x_data, dic)
    tags = [x[1] for x in x_data]
    onehot = make_tag_onehot(tags, tag_index)
    # print(word_embd.shape)
    # print(onehot.shape)
    X_data = np.append(word_embd, onehot, axis=1)
    # print(X_data.shape)
    return X_data

if __name__ == '__main__':
    # Args
    # eg:
    #   NTUSD-Fin
    #       python .\6.py .\data\training_set.json .\data\test_set.json .\data\outside_data\NTUSD-Fin\NTUSD_Fin_word_v1.0.json .\output\output.csv
    #   word2vec
    #       python .\6.py .\data\training_set.json .\data\test_set.json .\data\outside_data\GoogleNews-vectors-negative300.bin .\output\output.csv
    #   GLOVE
    #       python .\6.py .\data\training_set.json .\data\test_set.json .\data\outside_data\glove.6B\glove.6b.300d.txt .\output\output.csv
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    sdfin_path = sys.argv[3]
    gg_w2v_path = sys.argv[3]
    glove_path = sys.argv[3]
    output_path = sys.argv[4]

    x_train, y_train = read_train_test(train_path, [
        # 'sentiment'
        'snippet',
        'target',
        # 'tweet',
    ])
    x_test, y_test = read_train_test(test_path, [
        # 'sentiment'
        'snippet',
        'target',
        # 'tweet',
    ])

    # sdfin = read_sdfin(sdfin_path, [
    #     # 'bear_cfidf',
    #     # 'bear_freq',
    #     # 'bull_cfidf',
    #     # 'bull_freq',
    #     # 'chi_squared',
    #     # 'market_sentiment',
    #     'token',
    #     'word_vec',
    # ])
    # sdfin_dict = sdfin.set_index('token')['word_vec'].to_dict()
    glove_dic = gensim.models.KeyedVectors.load_word2vec_format(gg_w2v_path, binary=True)

    tags = [x[1] for x in x_train] + [x[1] for x in x_test]
    tag_index = make_tag_index(tags, thres=1)
    X_train = preprocess_on_x(x_train, glove_dic, tag_index)
    X_test = preprocess_on_x(x_test, glove_dic, tag_index)

    print(X_train.shape)
    print(X_test.shape)

    reg = RandomForestRegressor(n_estimators=500, max_depth=5)
    reg.fit(X_train, y_train.ravel())
    pred = reg.predict(X_test)

    error = metrics.mean_squared_error(pred, y_test[:,0])
    print('MSE: {}'.format(error))
    np.savetxt(output_path, pred, delimiter=',')
