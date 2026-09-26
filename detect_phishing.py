"""Step 3 of 3 -- the detector: persuasion scores, a text embedding and PhishSense, combined.

Every email gets three kinds of features: the five persuasion scores from the step 2 models,
a sentence embedding of its cleaned text, and PhishSense's phishing prediction on its raw
text. PyCaret picks and fits the final classifier. It is trained on benign mail and
traditional phishing only, and tested on the held-out set, which holds all the spear-phishing.
Every combination of the three feature kinds is run, so each one's contribution can be read off.

This is reference code. It shows the logic of the step; it is not a ready-to-run job.
"""
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from pycaret.classification import compare_models, finalize_model, predict_model, setup
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

PRINCIPLES = ["authority", "social_proof", "liking_similarity_deception",
              "commitment_reciprocation_consistency", "distraction"]
PPSE_COLS = [f"{p}_regressor" for p in PRINCIPLES]

EMBEDDING_MODELS = {"distilbert": "sentence-transformers/distilbert-base-nli-mean-tokens",   # 768 dims
                    "minilm": "sentence-transformers/all-MiniLM-L12-v2"}                    # 384 dims
PHISHSENSE = "AcuteShrewdSecurity/Llama-Phishsense-1B"

N_SPLITS = 5
RANDOM_STATE = 42


# ----------------------------------------------------------------------- features
def embed(texts, model_name: str) -> np.ndarray:
    """Mean-pooled sentence embedding of the cleaned text."""
    return SentenceTransformer(model_name).encode(list(texts), batch_size=64, convert_to_numpy=True)


def phishsense(raw_texts) -> np.ndarray:
    """PhishSense's verdict on the raw email, as 1 (phishing) or 0 (benign)."""
    tokenizer = AutoTokenizer.from_pretrained(PHISHSENSE)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(PHISHSENSE), PHISHSENSE)
    verdicts = []
    for text in raw_texts:
        prompt = ("Classify the following text as phishing or not. "
                  f"Respond with 'TRUE' or 'FALSE':\n\n{text}\nAnswer:")
        with torch.no_grad():
            out = model.generate(**tokenizer(prompt, return_tensors="pt"),
                                 max_new_tokens=5, temperature=0.01, do_sample=False)
        answer = tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[1].strip()
        verdicts.append(int(answer.upper().startswith("TRUE")))
    return np.array(verdicts)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phishsense"] = phishsense(df["email_text"])
    for name, model_name in EMBEDDING_MODELS.items():
        vectors = embed(df["clean_text"], model_name)
        df[[f"{name}_{i}" for i in range(vectors.shape[1])]] = vectors
    return df


def feature_combinations(df: pd.DataFrame) -> dict:
    """{embedding or none} x {PPSE on/off} x {PhishSense on/off}, minus the empty one: 11 in all."""
    combos = {}
    for emb in [None, *EMBEDDING_MODELS]:
        emb_cols = [c for c in df.columns if emb and c.startswith(f"{emb}_") and c[len(emb) + 1:].isdigit()]
        for use_ppse in (False, True):
            for use_phish in (False, True):
                cols = (PPSE_COLS if use_ppse else []) + (["phishsense"] if use_phish else []) + emb_cols
                if cols:
                    name = "_".join(n for n, on in [("ppse", use_ppse), ("phishsense", use_phish), (emb, emb)] if on)
                    combos[name] = cols
    return combos


# ----------------------------------------------------------------------- classifier
def select_and_fit(train: pd.DataFrame, features: list):
    """PyCaret compares its model library inside each of 5 folds (folds balanced by source
    corpus, ranked by F1). The fold winner that scores best on its own validation part is
    refit on the whole training set. The held-out set is never seen here."""
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best, best_f1 = None, -1.0
    for fit_idx, val_idx in folds.split(train, train["source"].factorize()[0]):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        setup(data=fit[features + ["label"]], target="label", session_id=RANDOM_STATE,
              preprocess=False, fix_imbalance=True, fold=3, html=False, verbose=False)
        model = compare_models(sort="F1", verbose=False)
        pred = predict_model(model, data=val[features], verbose=False)["prediction_label"]
        f1 = f1_score(val["label"], pred, average="macro")
        if f1 > best_f1:
            best, best_f1 = model, f1
    return finalize_model(best)


def evaluate(model, test: pd.DataFrame, features: list) -> dict:
    pred = predict_model(model, data=test[features], raw_score=True, verbose=False)
    y = test["label"].to_numpy()
    y_hat, p = pred["prediction_label"].to_numpy(), pred["prediction_score_1"].to_numpy()
    spear = test["source"].str.contains("spear-phishing").to_numpy()
    return {"accuracy": accuracy_score(y, y_hat),
            "precision": precision_score(y, y_hat),
            "recall": recall_score(y, y_hat),
            "f1": f1_score(y, y_hat),
            "roc_auc": roc_auc_score(y, p),
            "spear_phishing_detection_rate": y_hat[spear].mean()}


if __name__ == "__main__":
    corpus = add_features(pd.read_csv("corpus_with_regressor_scores.csv"))   # written by step 2
    train, test = corpus[corpus["split"] == "train"], corpus[corpus["split"] == "test"]

    rows = []
    for name, features in feature_combinations(corpus).items():
        model = select_and_fit(train, features)
        rows.append({"features": name, **evaluate(model, test, features)})
    pd.DataFrame(rows).to_csv("detection_results.csv", index=False)
