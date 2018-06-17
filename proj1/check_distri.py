import numpy as np
import pandas as pd
import sys

def read_train_test (path, cols):
    data = pd.read_json(path)
    x_data = data[cols].values
    y_data = data[['sentiment']].values
    return x_data, y_data

if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

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

    y = np.append(y_train, y_test)

    print('positive / neutral / negative')

    for thres in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

        print('thres:', thres)
        print('{} / {} / {}'.format((y > thres).sum(), np.logical_and(y < thres, y > -thres).sum(), (y < -thres).sum()))
