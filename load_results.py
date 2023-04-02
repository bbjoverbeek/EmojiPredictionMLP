import pickle
import sys

import datasets
from sklearn.metrics import classification_report


def main(argv):
    """Loads the results for a neural model from a pickle file

    Run with pickel as argument: $ python3 load_results.py predictions.pickle
    """

    # check if predictions were provided
    if len(argv) < 2:
        print('Attach a predictions file!', file=sys.stderr)
        sys.exit(1)

    # open predictions
    with open(argv[1], 'rb') as inp:
        predictions = pickle.load(inp)

    # get golden labels
    dataset = datasets.load_dataset('tweet_eval', 'emoji')

    # print classification report
    print(f'\nClassification report for {argv[1].split("_")[1]}:\n')
    print(classification_report(dataset['test']['label'], predictions))


if __name__ == '__main__':
    main(sys.argv)
