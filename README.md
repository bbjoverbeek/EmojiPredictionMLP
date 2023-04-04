# EmojiPredictionMLP
EmojiPrediction project for MLP

## Data
Dataset is from [Hugging Face](https://huggingface.co/datasets/tweet_eval)

## Running the models

Create a virtual env:

`$ python3 -m venv env`

`$ source env/bin/activate`

`$ python3 -m pip install -r requirements.txt`

Pick a model to train:

### Linear SVC trained on BERTweet embeddings 

1. Extract word embeddings from BERTweet by running [bertweet_embedding_extractor](./bertweet_embedding_extractor.ipynb), preferably on Google Collab, and following its instructions to download the pickle files to the ./data directory.

1. Run [svc.py](./svc.py): `$ python3 svc.py data/train_cls.pickle data/test_cls.pickle`


### MLP trained on BERTweet embeddings

1. Extract word embeddings from BERTweet by running [bertweet_embedding_extractor](./bertweet_embedding_extractor.ipynb), preferably on Google Collab, and following its instructions to download the pickle files to the ./data directory.

1. Run [mlp.py](./mlp.py): `$ python3 mlp.py data/train_cls.pickle data/test_cls.pickle`

_Optimal hyperparameters_  
To find the optimal hyperparameters for the MLP model we applied Grid Search. The code for this test can be found in [./mlp_gs.py](./mlp_gs.py).


### Fine-tune BERTweet for downstream classification task


