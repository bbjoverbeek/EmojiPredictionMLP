import pickle
import sys
import datasets
import time
import os
from pprint import pprint

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main(argv):
    """Runs a MLP classifier on BERTweet context vectors to predict emojis
    (Multi-layer Perceptron)

    Optimizes the MLP parameters with Grid Search

    To run: $ python3 mlp_gs.py data/train_cls.pickle data/test_cls.pickle

    """

    # check if the context vector files are provided
    if len(argv) < 3:
        print('add the context vector files', file=sys.stderr)
        sys.exit(1)

    # open the files
    with open(argv[1], 'rb') as inp:
        train_cls = pickle.load(inp)

    with open(argv[2], 'rb') as inp:
        test_cls = pickle.load(inp)

    # load the dataset for the corresponding labels
    dataset = datasets.load_dataset('tweet_eval', 'emoji')

    # initialize classifier
    max_iter = 1000

    mlp = MLPClassifier(max_iter=max_iter, verbose=True)

    parameters = {
        'hidden_layer_sizes':
            [(512, 512), (1028, 512), (512,), (512, 256, 512)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.001, 0.025, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    # train find the optimal hyperparameters and train the classifier
    start_time = time.time()

    clf = GridSearchCV(
        mlp, parameters, scoring='f1_macro', n_jobs=-1, cv=2, verbose=5
    )
    clf.fit(train_cls, dataset['train']['label'])

    print(
        f'\nThe MLP classifier took {(time.time() - start_time)/60:.2f} '
        f'minutes to optimize and train'
    )

    # predict on test set
    predictions = clf.predict(test_cls)

    print('The following parameters worked the best:')
    pprint(clf.best_params_)

    # print classification report
    print('\nClassification report:\n')
    print(classification_report(dataset['test']['label'], predictions))

    # create required directorie(s)
    os.makedirs(os.path.dirname('./predictions/'), exist_ok=True)

    # write predictions to a file
    with open('./predictions/predictions_mlp_gs.pickle', 'wb') as outp:
        pickle.dump(predictions, outp)


if __name__ == '__main__':
    main(sys.argv)
