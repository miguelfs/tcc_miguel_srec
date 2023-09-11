import pandas as pd
import pickle
import os
from tqdm import tqdm

def reorder(df, required_columns: list) -> pd.DataFrame:
    df = df.sort_values(by=required_columns)
    return df


def generate_map(df, required_columns):
    ids_map = {}
    for column in tqdm(required_columns, desc="Processing columns"):
        unique_ids = df[column].unique()
        ids_map.update({id: i for i, id in enumerate(unique_ids)})
    return ids_map


def preprocess_stuff():
    df = pd.read_csv(os.path.join('data', 'inda', 'raw', 'inda_barebone.csv'))
    required_columns = ['session_id', 'owner_id', 'user_id']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Required columns are missing from the DataFrame")
    
    df = reorder(df, required_columns)
    ids_map = generate_map(df, required_columns)
    with open(os.path.join('data', 'inda', 'raw', 'ids_map.pkl'), 'wb') as f:
        pickle.dump(ids_map, f)
    print('replacing')
    df = df.replace(ids_map)
    print('saving')
    df.to_csv(os.path.join('data', 'inda', 'raw', 'inda.csv'), index=False)

if __name__ == "__main__":
    preprocess_stuff()
