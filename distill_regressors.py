"""Step 2 of 3 -- the students: five small models learn to reproduce the teacher's scores.

For each PPSE principle, ModernBERT-large is fine-tuned as a regressor: it reads the cleaned
email text and predicts the score the LLM gave in step 1. Once trained, the five models score
any email with no LLM call. Their scores are the persuasion features of step 3.

This is reference code. It shows the logic of the step; it is not a ready-to-run job.
"""
import re

import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

PRINCIPLES = ["authority", "social_proof", "liking_similarity_deception",
              "commitment_reciprocation_consistency", "distraction"]

BACKBONE = "answerdotai/ModernBERT-large"
MAX_LENGTH = 512
VAL_SHARE = 0.2
SPLIT_SEED = 42
SEED = 44
EARLY_STOPPING_PATIENCE = 3

# Per-principle hyperparameters, from the Optuna search.
HYPERPARAMETERS = {
    "authority":                            dict(learning_rate=3.2016328345418156e-05, num_train_epochs=4, weight_decay=0.14754846656409662),
    "liking_similarity_deception":          dict(learning_rate=1.4786663677736627e-05, num_train_epochs=4, weight_decay=0.1583317760892298),
    "commitment_reciprocation_consistency": dict(learning_rate=5.408820239370336e-05, num_train_epochs=5, weight_decay=0.23115338858619786),
    "distraction":                          dict(learning_rate=1.53850275832709e-05, num_train_epochs=7, weight_decay=0.13984897016727754),
    "social_proof":                         dict(learning_rate=6.062870947030977e-05, num_train_epochs=7, weight_decay=0.13599754053809684),
}

# ----------------------------------------------------------------------- text cleaning
STOPWORDS = set(stopwords.words("english"))
HTML_TAG = re.compile(r"<[^>]+>")
KEEP = set("!'(),-.:;?_")


def preprocess(raw: str) -> str:
    tokens = word_tokenize(HTML_TAG.sub(" ", str(raw)).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join("".join(c for c in t if c.isalnum() or c in KEEP) for t in tokens)


# ----------------------------------------------------------------------- training
tokenizer = AutoTokenizer.from_pretrained(BACKBONE)


def to_dataset(texts, targets=None) -> Dataset:
    data = {"text": list(texts)}
    if targets is not None:
        data["label"] = np.asarray(targets, dtype=np.float32)
    return Dataset.from_dict(data).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True, remove_columns=["text"])


def spearman(pred) -> dict:
    """Rank agreement with the teacher: 0 = unrelated, 1 = same order. Picks the best epoch."""
    return {"spearman": spearmanr(pred.label_ids.ravel(), pred.predictions.ravel()).correlation}


def split_fit_val(train: pd.DataFrame):
    """80/20 inside each source corpus, so every source appears in both parts."""
    fit, val = [], []
    for _, group in train.groupby("source"):
        a, b = train_test_split(group, test_size=VAL_SHARE, random_state=SPLIT_SEED)
        fit.append(a)
        val.append(b)
    return pd.concat(fit), pd.concat(val)


def train_regressor(fit: pd.DataFrame, val: pd.DataFrame, principle: str, out_dir: str) -> Trainer:
    """One principle: a single output unit under squared-error loss."""
    args = TrainingArguments(
        output_dir=out_dir, **HYPERPARAMETERS[principle],
        per_device_train_batch_size=2, per_device_eval_batch_size=4,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="spearman", greater_is_better=True,
        report_to="none", seed=SEED)
    trainer = Trainer(
        model_init=lambda: AutoModelForSequenceClassification.from_pretrained(
            BACKBONE, num_labels=1, problem_type="regression"),
        args=args,
        train_dataset=to_dataset(fit["clean_text"], fit[principle]),
        eval_dataset=to_dataset(val["clean_text"], val[principle]),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=spearman,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)])
    trainer.train()
    trainer.save_model(out_dir)
    return trainer


def score(trainer: Trainer, texts) -> np.ndarray:
    """One forward pass per email; the regression output is the score."""
    return trainer.predict(to_dataset(texts)).predictions.ravel()


if __name__ == "__main__":
    corpus = pd.read_csv("ppse_intensity_corpus.csv")
    corpus["clean_text"] = corpus["email_text"].map(preprocess)
    fit, val = split_fit_val(corpus[corpus["split"] == "train"])

    for p in PRINCIPLES:
        trainer = train_regressor(fit, val, p, out_dir=f"regressors/{p}")
        corpus[f"{p}_regressor"] = score(trainer, corpus["clean_text"])

    corpus.to_csv("corpus_with_regressor_scores.csv", index=False)
