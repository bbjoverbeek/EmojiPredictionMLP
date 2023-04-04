import pickle
import sys
import datasets
import time
import os

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def main(argv):
    """Runs a Linear SVM on BERTweet context vectors to predict emojis

    Run like this: $ python3 svc.py data/train_cls.pickle data/test_cls.pickle

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
    max_iter = 10000
    dual = False
    classifier = LinearSVC(max_iter=max_iter, dual=dual)

    # train classifier
    start_time = time.time()

    # run top for a test on subset
    # classifier.fit(train_cls[:2000], dataset['train']['label'][:2000])
    classifier.fit(train_cls, dataset['train']['label'])

    print(
        f'\nThe SVC took {(time.time() - start_time)/60:.2f} minutes to train '
        f'and it has gone through {classifier.n_iter_} iterations to converge.'
    )

    # predict on test set
    predictions = classifier.predict(test_cls)

    # print classification report
    print('\nClassification report:\n')
    print(classification_report(dataset['test']['label'], predictions))

    # save predictions
    filename = (
        f'predictions_linearsvc_maxiter={max_iter}_'
        f'niter={classifier.n_iter_}_dual={dual}.pickle'
    )

    # create required directorie(s), if not present
    os.makedirs(os.path.dirname('./predictions/'), exist_ok=True)

    # write predictions to file
    with open(f'./predictions/{filename}', 'wb') as outp:
        pickle.dump(predictions, outp)


if __name__ == '__main__':
    main(sys.argv)
