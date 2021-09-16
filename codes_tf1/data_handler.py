
import pandas as pd


def load_tsv(filepath, columns):

    df = pd.read_csv(filepath, delimiter='\t')
    df = df[columns]
    df = df.fillna('').astype(str)

    return df

