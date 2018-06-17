# executed on Windows
# python .\7.py [training set] [testing set] [Google word2vec] [output csv path]
# eg: python .\7.py .\data\training_set.json .\data\test_set.json .\data\outside_data\GoogleNews-vectors-negative300.bin .\output\output.csv

import nltk
import numpy as np
import pandas as pd
import sys # argv
import xgboost as xgb

from collections import Counter
from gensim.models import KeyedVectors
from sklearn import metrics, ensemble

def read_train_test (path, cols):
    """ Read training set or testing set. """
    data = pd.read_json(path)
    x_data = data[cols].values
    y_data = data[['sentiment']].values
    return x_data, y_data

def read_word2vec (path):
    """ Read Google Word2vec. """
    # print('Reading Word2vec.')
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    # print('Finish reading Word2vec.')
    return word2vec

def sentence_to_embds (sentence, dic):
    """ Transform a sentence to a vector. """
    embds = []
    tokens = nltk.word_tokenize(sentence)
    for t in tokens:
        if t in dic:
            embds += [dic[t]]
    return np.array(embds)

def make_embds (x_data, dic):
    """ Transform all sentences in training set or testing set to a embedding matrix. """
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

        if embds.size > 0:
            new_embd = np.append(embds.max(axis=0), embds.min(axis=0))
            new_embd = np.append(new_embd, embds.mean(axis=0))
            new_embd = new_embd.reshape((1, 900))
            X_data = np.append(X_data, new_embd, axis=0)
        else:
            X_data = np.append(X_data, np.zeros((1, 900)), axis=0)
    return X_data[1:]

def make_tag_index (tags, thres=0):
    """ Create a one-to-one mapping from tags to integers. """
    tags_cnt = Counter(tags)
    tags_thres = [t if tags_cnt[t] > thres else '$OTHER' for t in tags]
    tags_thres_uni = list(set(tags_thres))
    tag_index = {t : tags_thres_uni.index(t) for t in tags_thres_uni}
    return tag_index

def make_tag_onehot (tags, tag_index):
    """ Transform tags to one-hot encoding. """
    onehot = np.zeros((len(tags), len(tag_index)))
    for i,t in enumerate(tags):
        if t in tag_index:
            onehot[i][tag_index[t]] = 1
        else:
            onehot[i][tag_index['$OTHER']] = 1
    return onehot

def preprocess_on_x (x_data, dic, tag_index):
    """ Preprocess on features.
        That is, concatenate the sentence embedding and the one-hot encoding.
    """
    word_embd = make_embds(x_data, dic)
    tags = [x[1] for x in x_data]
    onehot = make_tag_onehot(tags, tag_index)
    X_data = np.append(word_embd, onehot, axis=1)
    return X_data

def classify (y_data, thres=0.3):
    """ Classify datas by sentiment scores. """
    y_data_class = np.zeros_like(y_data)
    y_data_class[y_data > thres] = 1
    y_data_class[y_data < -thres] = -1
    return y_data_class

if __name__ == '__main__':
    """ Specify paths to data. """
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    w2v_path = sys.argv[3]
    output_path = sys.argv[4]

    """ Read data. """
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
    w2v_dic = read_word2vec(w2v_path)

    """ model 1: xgboost. """
    tags = [x[1] for x in x_train] + [x[1] for x in x_test]
    tag_index = make_tag_index(tags)
    X_train = preprocess_on_x(x_train, w2v_dic, tag_index)
    X_test = preprocess_on_x(x_test, w2v_dic, tag_index)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=20, show_stdv=False)
    # print('best num_boost_rounds = ', len(cv_output))
    # num_boost_rounds = len(cv_output)
    num_boost_rounds = 135
    model1 = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
    pred1 = model1.predict(dtest)

    """ model 2: gradient boosting. """
    tags = [x[1] for x in x_train] + [x[1] for x in x_test]
    tag_index = make_tag_index(tags, 2)
    X_train = preprocess_on_x(x_train, w2v_dic, tag_index)
    X_test = preprocess_on_x(x_test, w2v_dic, tag_index)

    model2 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3)
    model2.fit(X_train, y_train[:,0])
    pred2 = model2.predict(X_test)

    """ weighted ensemble. """
    pred = (pred1 * 3 + pred2) / 4

    """ Evaluate by MSE. """
    error = metrics.mean_squared_error(pred, y_test[:,0])
    print('MSE: {}'.format(error))

    """ Evaluate by F1-score. """
    y_test_class = classify(y_test[:,0])
    pred_class = classify(pred, 0.15)
    macro = metrics.f1_score(y_test_class, pred_class, average = 'macro')
    micro = metrics.f1_score(y_test_class, pred_class, average = 'micro')
    print('Micro F1: {}'.format(micro))
    print('Macro F1: {}'.format(macro))

    """ Save the predictions. """
    np.savetxt(output_path, pred, delimiter=',')
