import os
import pandas_gbq


def read_sql_file(project_id, db_name) -> str:
    project_id_formatted = f'`{project_id}`'
    path = os.path.join('inda', 'queries', 'raw_tracks_data.sql')
    with open(path, 'r') as file:
        query = file.read()
    return query.replace('${project_id}', project_id_formatted).replace('${db_name}', db_name)


def get_credentials():
    return os.environ.get('INDA_BIGQUERY_PROJECT_ID'), os.environ.get('INDA_BIGQUERY_DB_NAME')


def get_output_path():
    output_path = os.path.join('data', 'inda', 'raw', 'inda_barebone.csv')
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def get_tracks_data():
    project_id, db_name = get_credentials()
    output_path = get_output_path()

    query = read_sql_file(project_id, db_name)
    project_id = os.environ.get('INDA_BIGQUERY_PROJECT_ID')

    df = pandas_gbq.read_gbq(query, project_id)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    get_tracks_data()
