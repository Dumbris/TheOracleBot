import pandas as pd

def load(filename):
    return pd.read_json(filename, lines=True)