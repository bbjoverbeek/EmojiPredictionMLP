from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate
import numpy as np


def tokenize(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


seed = 42

dataset = load_dataset("tweet_eval", "emoji")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")


dataset = dataset.map(tokenize, batched=True)

small_train = dataset["train"].shuffle(seed=seed).select(range(1000))
small_eval = dataset["validation"].shuffle(seed=seed).select(range(1000))

model = AutoModelForSequenceClassification \
    .from_pretrained("vinai/bertweet-base", num_labels=20)

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch"
)

metric = evaluate.load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics
)

trainer.train()
