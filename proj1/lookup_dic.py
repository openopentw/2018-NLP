import nltk
import numpy as np
import pandas as pd
from sklearn import *
import sys # argv
import xgboost as xgb

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

if __name__ == '__main__':
    sdfin_path = sys.argv[1]
    gg_w2v_path = sys.argv[1]
    glove_path = sys.argv[1]

    # sdfin = read_sdfin(sdfin_path, [
    #     # 'bear_cfidf',
    #     # 'bear_freq',
    #     # 'bull_cfidf',
    #     # 'bull_freq',
    #     # 'chi_squared',
    #     'market_sentiment',
    #     'token',
    #     'word_vec',
    # ])
    # sdfin_dict = sdfin.set_index('token')['word_vec'].to_dict()
    glove = read_glove(glove_path)
    dic = glove

    sentence = 'Pivotal sees 27% upside for Alphabet'
    sentence = "I'll take the other side of that trade"

    tokens = nltk.word_tokenize(sentence)
    for t in tokens:
        if t in dic:
            print('{}: {}'.format(t, dic[t][:3]))
        else:
            print('{}: Not Found'.format(t))
