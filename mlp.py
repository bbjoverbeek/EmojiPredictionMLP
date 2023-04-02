import pickle
import sys
import datasets
import time
import os

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main(argv):
    """Runs a MLP classifier on BERTweet context vectors to predict emojis
    (Multi-layer Perceptron)

    Run like this: $ python3 mlp.py data/train_cls.picke data/test_cls.pickle

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
    solver = 'adam'
    activation = 'relu'
    hidden_layers = (512,)
    max_iter = 1000

    classifier = MLPClassifier(
        solver=solver, activation=activation, hidden_layer_sizes=hidden_layers,
        max_iter=max_iter
    )

    # train classifier
    start_time = time.time()

    # classifier.fit(train_cls[:1000], dataset['train']['label'][:1000])
    classifier.fit(train_cls, dataset['train']['label'])

    print(
        f'\nThe MLP classifier took {(time.time() - start_time)/60:.2f} '
        f'minutes to train '
        f'and it has gone through {classifier.n_iter_} iterations to converge.'
    )

    # predict on test set
    predictions = classifier.predict(test_cls)

    # print classification report
    print('\nClassification report:\n')
    print(classification_report(dataset['test']['label'], predictions))

    # save predictions
    filename = (
        f'predictions_mlpclassifier_'
        f'layers={"-".join([str(num) for num in hidden_layers])}_'
        f'niter={classifier.n_iter_}_solver={solver}_'
        f'activation={activation}.pickle'
    )

    # create required directorie(s)
    os.makedirs(os.path.dirname('./predictions/'), exist_ok=True)

    with open(f'./predictions/{filename}', 'wb') as outp:
        pickle.dump(predictions, outp)


if __name__ == '__main__':
    main(sys.argv)
