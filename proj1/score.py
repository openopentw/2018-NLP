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
