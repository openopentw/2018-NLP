""" Train and test. """

import sys #argv
# import json
import nltk
import numpy as np
import pandas as pd
import xgboost as xgb

from gensim.models import KeyedVectors
from sklearn import metrics

import bt2us

MISSING_VAL = -999

XGB_PARAMS = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'multi:softmax',
    'num_class': 19,
    'eval_metric': 'mlogloss',
    'silent': 0
}

RELS = {
    'cause-effect': 0,
    'instrument-agency': 2,
    'product-producer': 4,
    'content-container': 6,
    'entity-origin': 8,
    'entity-destination': 10,
    'component-whole': 12,
    'member-collection': 14,
    'message-topic': 16,
    'other': 18,
}

def get_entity(sen):
    """ Get the two entities labeled in the given string.

    Args:
        string: the string that contains <e1></e1> and <e2></e2> that
            identify the entity.

    Return: Two strings that are the two entities from the given string.

    """
    e_1 = sen.split('<e1>')[1].split('</e1>')[0]
    e_2 = sen.split('<e2>')[1].split('</e2>')[0]
    e_1 = bt2us.trans_term(e_1)
    e_2 = bt2us.trans_term(e_2)
    return e_1, e_2

def get_rel(rel_str):
    """ Get the relation labeled in the given string.

    Args:
        rel_str: the string that contains the string that identify the
            relation.

    Return: A number that correspond to that relation (specified in
        dict RELS).

    """
    rel = rel_str.split('(')[0]
    order = RELS[rel]
    if rel != 'other':
        e_precede = rel_str.split('(')[1].split(',')[0]
        if e_precede == 'e2':
            order += 1
    return order

def read_train(path):
    """ Read training data. """
    with open(path) as infile:
        data = infile.read()
    data = data.lower()
    data = data.split('\n\n')[:-1]

    x_data = []
    y_data = []
    for dat in data:
        dat = dat.split('\n')[:2]
        sen = dat[0].split('\t')[1][1:-1]

        e_1, e_2 = get_entity(sen)
        order = get_rel(dat[1])

        x_data += [{
            'e1': e_1,
            'e2': e_2,
        }]
        y_data += [order]
    return x_data, y_data

def read_test(path):
    """ Read testing data. """
    with open(path) as infile:
        data = infile.read()
    data = data.lower()
    data = data.split('\n')
    if data[-1] == '':
        data = data[:-1]

    x_data = []
    for dat in data:
        sen = dat.split('\t')[1][1:-1]

        e_1, e_2 = get_entity(sen)

        x_data += [{
            'e1': e_1,
            'e2': e_2,
        }]
    return x_data

def read_ans(path):
    """ Read answer keys. """
    with open(path) as infile:
        data = infile.read()
    data = data.lower()
    data = data.split('\n')
    if data[-1] == '':
        data = data[:-1]

    y_data = []
    for dat in data:
        rel_str = dat.split('\t')[1]
        order = get_rel(rel_str)
        y_data += [order]
    return y_data

def read_word2vec(path):
    """ Read Google Word2Vec. """
    print('Reading Word2vec.')
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    print('Finish reading Word2vec.')
    return word2vec

def read_glove(path):
    """ Read GLOVE. """
    print('Reading GLOVE.')
    glove = {}
    with open(path, encoding='utf8') as infile:
        for line in infile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    print('Finish reading GLOVE.')
    return glove

def read_sdfin(path, cols):
    """ Read NTUSD-Fin. """
    print('Reading NTUSD-Fin.')
    data = pd.read_json(path)
    data = data[cols]
    print('Finish reading NTUSD-Fin.')
    return data

def term_to_embd(term, dic):
    """ Transform a term to a d-dim embd.

    Args:
        term: A string that contain one or more words.
        dic: a dict file that maps a term to a vector.

    Returns:
        ret_embd: a d-dim numpy-array, where d is the dimension of the embedding
            vector from dic.
        not_in_cnt: the number of terms that are not in dic.

    """
    not_in_cnt = 0
    if term in dic:
        ret_embd = np.array(dic[term])
    else:
        embds = []
        tokens = nltk.word_tokenize(term)
        for token in tokens:
            if token in dic:
                embds += [dic[token]]
            else:
                print('missing: "{}"'.format(token))
                not_in_cnt += 1
        if embds:
            ret_embd = np.array(embds).mean(axis=0)
        else:
            ret_embd = np.ones(300) * -999
    return ret_embd, not_in_cnt

def make_embds(x_data, dic):
    """ Transform given data to a (n * 600) vector.

    Args:
        X_data: A dict that contains keys 'e1' and 'e2'.
        dic: a dict file that maps a term to a vector.

    Returns:
        a (n * 600) numpy-array, where n is the number of term in x_data.

    """
    ret_data = np.zeros((8000, 600))
    not_in_cnt = 0
    for i, term in enumerate(x_data):
        embd1, new_cnt_1 = term_to_embd(term['e1'], dic)
        embd2, new_cnt_2 = term_to_embd(term['e2'], dic)
        not_in_cnt += new_cnt_1 + new_cnt_2
        ret_data[i, :] = np.append(embd1, embd2)
    print(not_in_cnt)
    return ret_data

def main():
    """ Main function. """
    assert len(sys.argv) > 1, 'Please give me the path to training file.'
    assert len(sys.argv) > 2, 'Please give me the path to testing file.'
    assert len(sys.argv) > 3, 'Please give me the path to answer key file.'
    assert len(sys.argv) > 4, 'Please give me the path to Google Word2Vec file.'

    x_train, y_train = read_train(sys.argv[1])
    test = read_test(sys.argv[2])
    y_test = read_ans(sys.argv[3])
    # json.dump(train[:10], sys.stdout, indent=2)
    w2v_dic = read_word2vec(sys.argv[4])
    # glove_dic = read_glove(sys.argv[4])
    # sdfin = read_sdfin(sys.argv[4], ['token', 'word_vec'])
    # sdfin_dic = sdfin.set_index('token')['word_vec'].to_dict()

    dic = w2v_dic
    x_train = make_embds(x_train, dic)
    x_test = make_embds(test, dic)
    print(x_train[:10])
    print(x_train.shape)
    print(x_test[:10])
    print(x_test.shape)

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    cv_output = xgb.cv(XGB_PARAMS,
                       dtrain,
                       num_boost_round=2000,
                       early_stopping_rounds=20,
                       verbose_eval=25,
                       show_stdv=True)
    print('best num_boost_rounds = ', len(cv_output))
    num_boost_rounds = len(cv_output)
    # num_boost_rounds = 1436
    model = xgb.train(XGB_PARAMS,
                      dtrain,
                      num_boost_round=num_boost_rounds)
    pred = model.predict(dtest)

    np.savetxt('a.csv', pred, delimiter=',')

    macro = metrics.f1_score(pred, y_test, average='macro')
    print('MACRO: {}'.format(macro))

if __name__ == '__main__':
    main()
