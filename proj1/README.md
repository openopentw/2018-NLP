# Proj1

## TODO

- [x] Add tag data. (maybe by using one-hot encoding)
- [ ] Calculate WCS score
- [ ] Calculate f1 score
- [ ] Try RNN (do padding first)
- [ ] Try [Microsoft / LightGBM](https://github.com/Microsoft/LightGBM)
- [ ] Try LIBSVM
- [ ] Try other methods. (e.g.: AdaBoostRegressor, RandomForest, ... etc)

## Experiment Result

| id   | MSE score  | method                       | outside_data                |
| ---- | ---------- | ---------------------------- | --------------------------- |
| rand | 0.5209     | random from [-1, 1]          |                             |
| zero | 0.1557     | all 0                        |                             |
| 1    | 0.1073     | sklearn: gradient boosting   | NTUSD-Fin: market_sentiment |
| 2    | 0.1018     | sklearn: gradient boosting   | NTUSD-Fin: word_vec         |
| 3    | 0.0920     | XGBoost                      | NTUSD-Fin: word_vec         |
| 4    | 0.0964     | XGBoost                      | GLOVE                       |
| 5    | **0.0918** | XGBoost, use tags as one-hot | NTUSD-Fin: word_vec         |
| 6    | 0.0957     | XGBoost, use tags as one-hot | GLOVE                       |

## Ensemble Result

| id   | MSE score  | used_id            | weight                     |
| ---- | ---------- | ------------------ | -------------------------- |
| 1    | 0.0884     | [1, 2, 3, 4]       | [1, 1, 1.5, 1.5]           |
| 2    | 0.0881     | [1, 2, 3, 4, 5]    | [1, 1, 1.5, 1.5, 1.5]      |
| 3    | 0.0865     | [1, 2, 3, 4, 5, 6] | [1, 1, 1.5, 1.5, 1.5, 1.5] |
| 4    | **0.0853** | [3, 4, 5, 6]       | [1.2, 1, 1.2, 1]           |

## References

- GitHub / task5: https://github.com/magizbox/underthesea/wiki/SemEval-2017-Task-5
- paper: http://nlp.arizona.edu/SemEval-2017/pdf/SemEval152.pdf


