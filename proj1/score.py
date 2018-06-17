import numpy as np
import pandas as pd
import sys # argv

from math import sqrt
from sklearn import metrics

def weighted_cosine_similarity (g, p):
    score = (output * y_test).sum()
    score /= sqrt((g * g).sum())
    score /= sqrt((p * p).sum())
    score = score if score > 0 else -score
    return score

def classify (y_data, thres=0.3):
    y_data_class = np.zeros_like(y_data)
    y_data_class[y_data > thres] = 1
    y_data_class[y_data < -thres] = -1
    return y_data_class

# Args
# e.g.
#    python .\score.py .\data\test_set.json .\output\5.csv WCS
test_path = sys.argv[1]
output_path = sys.argv[2]
metrics_str = sys.argv[3]

test = pd.read_json(test_path)
y_test = test[['sentiment']].values[:,0]

output = np.genfromtxt(output_path)

if metrics_str == 'MSE':
    error = metrics.mean_squared_error(output, y_test)
    print('MSE: {}'.format(error))
elif metrics_str == 'WCS':
    score = weighted_cosine_similarity(output, y_test)
    print('WCS: {}'.format(score))
elif metrics_str == 'F1':
    f1_list = []
    ij_list = []
    thres1 = 0.3
    for j in range(1, 100):
        thres2 = j / 100

        y_test_class = classify(y_test, thres1)
        pred_class = classify(output, thres2)
        macro = metrics.f1_score(y_test_class, pred_class, average = 'macro')
        micro = metrics.f1_score(y_test_class, pred_class, average = 'micro')

        ij_list += [(thres2, macro, micro)]
        f1_list += [macro * micro]

    thres2, macro, micro = ij_list[f1_list.index(max(f1_list))]
    print('thres2: {}'.format(thres2))
    print('Micro F1: {}'.format(micro))
    print('Macro F1: {}'.format(macro))
