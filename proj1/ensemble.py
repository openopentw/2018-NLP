import numpy as np
import pandas as pd
import sys
from sklearn import metrics

# Arg
# e.g.
#    python .\ensemble.py .\data\test_set.json .\output\ensemble.csv
test_path = sys.argv[1]
output_path = sys.argv[2]

# MSE: 0.08839117954201112
csv_list = [
    './output/7-0.csv',
    './output/7-gbr.csv',
]

weights = np.array([
    3,
    1,
])
weights = weights / weights.sum()

results = np.zeros((len(csv_list), 634))
for i,csv in enumerate(csv_list):
    vs = np.genfromtxt(csv)
    results[i] = vs
# print(results)
# print(results.shape)

result = weights.dot(results)

test = pd.read_json(test_path)
y_test = test[['sentiment']].values[:,0]

error = metrics.mean_squared_error(result, y_test)
print('MSE: {}'.format(error))
np.savetxt(output_path, result, delimiter=',')
