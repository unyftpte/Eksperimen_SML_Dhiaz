
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

COL_ITEM_ID="article_id"; COL_TITLE="title"; COL_CONTENT="content"; COL_CAT="category"
COL_USER="user_id"; COL_SCORE="interaction"

def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def run_pipeline(articles_path, inter_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    items = pd.read_csv(articles_path)
    inter = pd.read_csv(inter_path)

    items["text"] = (
        items.get(COL_TITLE, "").fillna('') + " " +
        items.get(COL_CONTENT, "").fillna('') + " " +
        (items.get(COL_CAT, "").fillna('') if COL_CAT in items.columns else "")
    ).map(normalize_text)

    items = items.drop_duplicates(subset=[COL_ITEM_ID]).dropna(subset=["text"])
    inter  = inter.dropna(subset=[COL_USER, COL_ITEM_ID, COL_SCORE])

    if pd.api.types.is_numeric_dtype(inter[COL_SCORE]):
        scaler = MinMaxScaler()
        inter["rating_scaled"] = scaler.fit_transform(inter[[COL_SCORE]])
    else:
        inter["rating_scaled"] = 1.0

    train, valid = train_test_split(
        inter[[COL_USER, COL_ITEM_ID, "rating_scaled"]],
        test_size=0.2, random_state=42, shuffle=True
    )

    _ = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2).fit_transform(items["text"])

    out_items = items[[COL_ITEM_ID, COL_TITLE]].copy()
    if COL_CAT in items.columns: out_items[COL_CAT] = items[COL_CAT]
    out_items["text"] = items["text"]

    out_items.to_csv(f"{out_dir}/items_clean.csv", index=False)
    train.to_csv(f"{out_dir}/interactions_train.csv", index=False)
    valid.to_csv(f"{out_dir}/interactions_valid.csv", index=False)

    print("✅ Preprocessing selesai.")
    print("Items :", out_items.shape, "| Train :", train.shape, "| Valid :", valid.shape)
    print("Output ->", os.path.abspath(out_dir))

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", default="./articles.csv")
    p.add_argument("--inter",    default="./interactions.csv")
    p.add_argument("--out_dir",  default="./namadataset_preprocessing")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args, _unknown = parser.parse_known_args()  # ignore -f dari Jupyter
    run_pipeline(args.articles, args.inter, args.out_dir)
