import argparse
import pandas as pd
import os
import pandas_gbq
from tqdm import tqdm


def get_credentials():
    return os.environ.get('INDA_BIGQUERY_PROJECT_ID'), os.environ.get('INDA_BIGQUERY_DB_NAME')


def get_session_map(project_id, db_name):
    query = read_session_query(project_id, db_name)
    df = pandas_gbq.read_gbq(query, project_id)
    session_map = {row['id']: row['name'] for _, row in df.iterrows()}
    owner_map = {row['id']: row['user_id'] for _, row in df.iterrows()}
    return session_map, owner_map


def read_session_query(project_id, db_name) -> str:
    project_id_formatted = f'`{project_id}`'
    path = os.path.join('inda', 'queries', 'session_map.sql')
    with open(path, 'r') as file:
        query = file.read()
    return query.replace('${project_id}', project_id_formatted).replace('${db_name}', db_name)


def read_user_query(project_id, db_name) -> str:
    project_id_formatted = f'`{project_id}`'
    path = os.path.join('inda', 'queries', 'user_map.sql')
    with open(path, 'r') as file:
        query = file.read()
    return query.replace('${project_id}', project_id_formatted).replace('${db_name}', db_name)


def get_user_map(project_id, db_name):
    query = read_user_query(project_id, db_name)
    df = pandas_gbq.read_gbq(query, project_id)
    return {row['id']: row['username'] for _, row in df.iterrows()}


def apply_labels(input_path):
    project_id, db_name = get_credentials()
    df = pd.read_csv(input_path)
    # get session id name map
    session_map, owner_map = get_session_map(project_id, db_name)
    # get user id name map
    user_map = get_user_map(project_id, db_name)
    # apply session id name map over the df
    # replace
    df['Owner'] = df[['SessionId']].replace(owner_map)
    # Remove the 'Owner' column from its current position
    owner_column = df.pop('Owner')
    # Insert the 'Owner' column as the leftmost column
    df.insert(0, 'Owner', owner_column)
    df['SessionId'] = df[['SessionId']].replace(session_map)
    # for all the columns that start with Recommendations@, replace the values
    for column in tqdm(df.columns, desc="Processing Recommendations@K columns"):
        if column.startswith('Recommendations@'):
            df[column] = df[[column]].replace(user_map)
    # create a column name Owner based on the SessionId column, getting the value from user_map
    df['Owner'] = df[['Owner']].replace(user_map)
    labeled_path = input_path.replace('.csv', '_labeled.csv')
    df.drop(columns=['Recommendations'], inplace=True)
    # move Scores column to the end
    scores_column = df.pop('Scores')
    df.insert(len(df.columns), 'Scores', scores_column)
    df.to_csv(labeled_path, index=False)


def do_recommendations_break(df, ids_map):
    # Split the "Recommendations" column into a list of values
    df['Recommendations'] = df['Recommendations'].str.split(',').apply(lambda x: [int(id) for id in x])

    # Determine the maximum number of elements in any list
    max_len = df['Recommendations'].apply(len).max()

    # Create new columns for each element in the list
    for i in tqdm(range(1, max_len), desc="Unwrapping Recommendations@K"):
        df[f'Recommendations@{i}'] = df['Recommendations'].str[i - 1] if i <= len(df['Recommendations'][0]) else None
        df[f'Recommendations@{i}'] = df[[f'Recommendations@{i}']].replace(ids_map)
    return df


def apply_map(input_file):
    # read the .csv file
    df = pd.read_csv(input_file, sep=';')
    df = df.head(100)
    # get map path
    map_path = os.path.join('data', 'inda', 'raw', 'ids_map.pkl')
    # retrieve python map from pickle file
    ids_map = pd.read_pickle(map_path)
    df[['SessionId']] = df[['SessionId']].replace(ids_map)
    df = do_recommendations_break(df, ids_map)
    # save the new .csv file with the same name but adding a suffix
    mapped_path = input_file.replace('.csv', '_mapped.csv')
    df.to_csv(mapped_path, index=False)
    return mapped_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This postprocess labels the session and user ids')
    parser.add_argument('-f', '--file', type=str, help='path to the input data', required=True)
    args = parser.parse_args()
    if not os.path.exists(args.file):
        raise Exception('File not found')
    output_path = apply_map(args.file)
    apply_labels(output_path)
