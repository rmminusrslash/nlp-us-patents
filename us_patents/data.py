import re
from pprint import pprint

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold
from transformers import PreTrainedTokenizer

from .cpc_texts import get_cpc_texts

# adopted from https://github.com/Gladiator07/U.S.-Patent-Phrase-to-Phrase-Matching-Kaggle/blob/main/src/data/dataset.py


def prepare_data(
    df: pd.DataFrame, cpc_scheme_xml_dir: str, cpc_title_list_dir: str
) -> pd.DataFrame:
    """
    Prepares data (applies both for `train` and `inference` mode)

    Args:
        train_df (pd.DataFrame): train dataframe
        cpc_scheme_xml_dir (str): directory where cpc_scheme_xml files are saved
        cpc_title_list_dir (str): directory where cpc_title_list files are saved

    Returns:
        pd.DataFrame: prepared `train_df`
    """

    cpc_texts = get_cpc_texts(
        cpc_scheme_xml_dir=cpc_scheme_xml_dir, cpc_title_list_dir=cpc_title_list_dir
    )
    df["context_text"] = df["context"].map(cpc_texts)

    df["cleaned_context_text"] = df["context_text"].map(
        lambda x: re.sub("[^A-Za-z0-9]+", " ", x)
    )

    # prepare input text
    sep = " [s] "
    df["input_text"] = (
        df["cleaned_context_text"].str.lower() + sep + df["anchor"] + sep + df["target"]
    )

    return df


def create_folds(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    print_summary: bool = False,
):
    """
    Split data grouped by anchor and stratified by score (`StratifiedGroupKFold`)

    Args:
        train_df (pd.DataFrame): train dataframe
        n_folds (int, optional): number of splits. Defaults to 5.
        seed (int, optional): random state. Defaults to 42.

    Returns:
        pd.DataFrame: dataframe with `fold` as column for splits
    """

    # CV SPLIT
    #######################################
    # grouped by anchor + stratified by score
    #######################################

    train_df["score_bin"] = pd.cut(train_df["score"], bins=5, labels=False)
    train_df["fold"] = -1
    logger.info(
        f"Prediction score bins: {train_df['score_bin'].value_counts().sort_index()}"
    )
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = sgkf.split(
        X=train_df,
        y=train_df["score_bin"].to_numpy(),
        groups=train_df["anchor"].to_numpy(),
    )
    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_df.loc[val_idx, "fold"] = fold
    train_df["fold"] = train_df["fold"].astype(int)

    # #######################################
    if print_summary:
        print("\nSamples per fold:")
        print(train_df["fold"].value_counts())

        print("\n Mean score per fold:")
        scores = [
            train_df[train_df["fold"] == f]["score"].mean() for f in range(n_folds)
        ]
        pprint(scores)

        print("\n Score distribution per fold:")

        [
            print(train_df[train_df["fold"] == f]["score"].value_counts())
            for f in range(n_folds)
        ]

    return train_df
