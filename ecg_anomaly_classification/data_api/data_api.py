import os
import ast
import wfdb
import pandas as pd


def read_ecg(ecg_path):
    return wfdb.rdsamp(ecg_path)[0]


def _load_all_data(ptb_path):
    csv_path = os.path.join(ptb_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df["target"] = df.scp_codes.apply(lambda x: 1 if "NORM" in x else 0)
    return df[["strat_fold", "filename_lr", "target"]]


def load_train_set(ptb_path):
    df = _load_all_data(ptb_path)
    return df[(df["strat_fold"] != 9) & (df["strat_fold"] != 10)]


def load_validation_set(ptb_path):
    df = _load_all_data(ptb_path)
    return df[df["strat_fold"] == 9]


def load_test_set(ptb_path):
    df = _load_all_data(ptb_path)
    return df[df["strat_fold"] == 10]
