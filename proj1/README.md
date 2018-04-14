# proj1

## TODO

- [ ] Add tag data. (maybe by using one-hot encoding)
- [ ] Calculate f1 score
- [ ] Try other methods. (e.g.: AdaBoostRegressor, RandomForest, ... etc)

## Experiment Result

| id   | score      | method                     | outside_data                |
| ---- | ---------- | -------------------------- | --------------------------- |
| rand | 0.5209     | random from [-1, 1]        |                             |
| zero | 0.1557     | all 0                      |                             |
| 1    | 0.1073     | sklearn: gradient boosting | NTUSD-Fin: market_sentiment |
| 2    | 0.1018     | sklearn: gradient boosting | NTUSD-Fin: word_vec         |
| 3    | **0.0920** | XGBoost                    | NTUSD-Fin: word_vec         |
| 4    | 0.0964     | XGBoost                    | GLOVE                       |

## Ensemble Result

| id   | score      | method                     | outside_data                |
| ---- | ---------- | -------------------------- | --------------------------- |
| 1    | **0.0884** | sklearn: gradient boosting | NTUSD-Fin: market_sentiment |

## References

- GitHub / task5: https://github.com/magizbox/underthesea/wiki/SemEval-2017-Task-5
- paper: http://nlp.arizona.edu/SemEval-2017/pdf/SemEval152.pdf

