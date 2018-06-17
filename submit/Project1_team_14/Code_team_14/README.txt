- Environment
    - Windows 10
    - Python 3.6

- Requirements
    - Data
        - Google Word2Vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

    - Python Modules
        - gensim
        - nltk
        - numpy
        - pandas
        - scikit-learn
        - xgboost

- Commands to Run the Code
    - Python final.py [training set] [testing set] [Google word2vec] [output csv path]
        - For example,
        - python .\final.py .\data\training_set.json .\data\test_set.json .\data\outside_data\GoogleNews-vectors-negative300.bin .\output\output.csv

    - Sample Output
        - MSE: ...
          Micro F1: ...
          Macro F1: ...
