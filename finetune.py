from TweetNormalizer import normalizeTweet
# clone bertweet repo into content folder
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

evaluate_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
seed = 45
dataset = load_dataset("tweet_eval", "emoji")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")


def tokenize(data):
    """
    Using the normalize tweet function from bertweet, so the text is optimized
    for the bertweet model.
    Then the text is tokenized and padded and truncated. The max length is set
    by the max length of the bertweet model.
    """
    return tokenizer(
        normalizeTweet(data["text"]), padding="max_length", truncation=True
    )


def compute_metrics(p):
    """
    This will be run after each epoch. It will calculate the accuracy, macro
    recall, macro precision and macro f1-score.
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


dataset = dataset.map(tokenize)
train = dataset["train"].shuffle(seed=seed)
eval = dataset["validation"].shuffle(seed=seed)
test = dataset["test"].shuffle(seed=seed)

print(train[0:5])

model = AutoModelForSequenceClassification \
    .from_pretrained("vinai/bertweet-base", num_labels=20)

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    learning_rate=0.00005,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0,
    num_train_epochs=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=eval,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("test_trainer")
