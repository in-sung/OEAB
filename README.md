## Adaptive boosting for ordinal target variables using neural networks

### Abstract
Boosting has proved its superiority, increasing the diversity of base classifiers, mainly in various classification problems. In reality, target variables in classification often are formed by numerical variables, in possession of ordinal information. However, existing boosting algorithms for classification are unable to reflect such ordinal target variables, resulting in non-optimal solutions. In this paper, we propose a novel algorithm of ordinal encoding adaptive boosting using a multi-dimensional encoding scheme for ordinal target variables. Extending an original binary-class adaptive boosting, the proposed algorithm is equipped with multi-class exponential loss function. We show that it achieves the Bayes classifier and establishes forward stagewise additive modeling. We demonstrate the performance of the proposed algorithm with a base learner as a neural network. Our experiments show that it outperforms existing boosting algorithms in various ordinal datasets.

### Run the test
'''
$ python main.py --dataname toy --n_estimators 10 --hidden_node 16
'''
