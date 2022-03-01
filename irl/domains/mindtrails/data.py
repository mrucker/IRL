import pandas as pd #type: ignore

from irl.config import DATA_PATH

def read_labeled_state() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/mindtrails/labeled_state.csv")

def read_action_log() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}/mindtrails/action_log.csv")