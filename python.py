import datasets
import pandas as pd

df = datasets.load_dataset("tweet_eval", "emoji")["train"].to_pandas()
with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.precision', 3,
):
    print(df.iloc[200])
