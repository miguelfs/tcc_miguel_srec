import pandas as pd
import pickle
import os
from tqdm import tqdm


def generate_map(df, required_columns):
    ids_map = {}
    index = 0
    for column in tqdm(required_columns, desc="Processing columns"):
        unique_ids = df[column].unique()
        for unique_id in tqdm(unique_ids, desc="Processing unique ids"):
            if unique_id not in ids_map.values():
                ids_map[index] = unique_id
                index += 1
    return ids_map


def preprocess_stuff():
    raw_folder = os.path.join('data', 'inda', 'raw')
    df = pd.read_csv(os.path.join(raw_folder, 'inda_barebone.csv'))
    required_columns = ['session_id', 'owner_id', 'user_id']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Required columns are missing from the DataFrame")

    ids_map = generate_map(df, required_columns)
    with open(os.path.join(raw_folder, 'ids_map.pkl'), 'wb') as f:
        pickle.dump(ids_map, f)
    print('replacing')
    reversed_map = {v: k for k, v in ids_map.items()}
    df[['session_id', 'owner_id', 'user_id']] = df[['session_id', 'owner_id', 'user_id']].replace(reversed_map)
    print('saving')
    df.to_csv(os.path.join(raw_folder, 'inda.csv'), index=False)


if __name__ == "__main__":
    preprocess_stuff()
