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
    return x_data, y_data[:,0]

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
    #       python .\1.py .\data\training_set.json .\data\test_set.json .\data\NTUSD-Fin\NTUSD_Fin_word_v1.0.json .\output\output.csv
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

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)

    # MSE: 
    # Macro F1 (thres=0.4): 
    # Micro F1 (thres=0.4): 
    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
    print('best num_boost_rounds = ', len(cv_output))
    num_boost_rounds = len(cv_output)
    # num_boost_rounds = 1436
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
    pred = model.predict(dtest)

    error = metrics.mean_squared_error(pred, y_test)
    print('MSE: {}'.format(error))
    np.savetxt(output_path, pred, delimiter=',')
