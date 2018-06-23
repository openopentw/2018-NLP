""" Train and test. """

import sys #argv
# import json
import nltk
import numpy as np
import pandas as pd
# import xgboost as xgb

from gensim.models import KeyedVectors

import bt2us

MISSING_VAL = -999

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

def read_train(path):
    """ Read training data. """
    with open(path) as infile:
        data = infile.read()
    data = data.lower()
    data = data.split('\n\n')[:-1]

    ret = []
    for dat in data:
        dat = dat.split('\n')[:2]
        sen = dat[0].split('\t')[1][1:-1]

        # get entities
        e_1, e_2 = get_entity(sen)

        # get relations
        rel_str = dat[1]
        rel = rel_str.split('(')[0]
        order = RELS[rel]
        if rel != 'other':
            e_precede = rel_str.split('(')[1].split(',')[0]
            if e_precede == 'e2':
                order += 1

        # push_back to ret
        ret += [{
            'e1': e_1,
            'e2': e_2,
            # 'rel_str': rel_str,
            'rel': order,
        }]
    return ret

def read_test(path):
    """ Read testing data. """
    with open(path) as infile:
        data = infile.read()
    data = data.lower()
    data = data.split('\n')

    ret = []
    for dat in data:
        sen = dat.split('\t')[1][1:-1]

        # get entities
        e_1, e_2 = get_entity(sen)

        # push_back to ret
        ret += [{
            'e1': e_1,
            'e2': e_2,
        }]
    return ret

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
                print(token)
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
    assert len(sys.argv) > 3, 'Please give me the path to Google Word2Vec file.'

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    w2v_path = sys.argv[3]
    # sdfin_path = sys.argv[3]
    # glove_path = sys.argv[3]

    train = read_train(train_path)
    test = read_train(test_path)
    # json.dump(train[:10], sys.stdout, indent=2)
    w2v_dic = read_word2vec(w2v_path)
    # glove_dic = read_glove(glove_path)
    # sdfin = read_sdfin(sdfin_path, ['token', 'word_vec'])
    # sdfin_dic = sdfin.set_index('token')['word_vec'].to_dict()
    dic = w2v_dic

    x_train = make_embds(train, dic)
    x_test = make_embds(test, dic)
    print(x_train[:10])
    print(x_test.shape)

if __name__ == '__main__':
    main()
