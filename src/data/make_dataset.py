import pandas as pd
from pathlib import Path
import click
import logging
from check_structure import check_existing_file, check_existing_folder
import os
from sklearn.preprocessing import StandardScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw_data) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_raw = f"{input_filepath}/raw.csv"
    output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    process_data(input_filepath_raw, output_filepath)

def process_data(input_filepath_raw, output_filepath):

    df_minerals = import_dataset(input_filepath_raw, sep=",", low_memory=False)
    df_minerals = convert_columns_type_date(df_minerals)

    X_train, X_test, y_train, y_test = split_data_time_series(df_minerals)
    #X_train, X_test = fill_nan_values(X_train, X_test)

    create_folder_if_necessary(output_filepath)
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

def drop_columns(df):
    list_to_drop = ['ave_flot_air_flow', 'ave_flot_level']
    list_to_drop = []
    df.drop(list_to_drop, axis=1, inplace=True)
    return df

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def convert_columns_type_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def drop_lines_with_nan_values(df):
    col_to_drop_lines = []
    df = df.dropna(subset=col_to_drop_lines, axis=0)
    return df

def split_data_time_series(df):
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    
    split_index = int(len(df) * 0.8)
    X_train, X_test = feats.iloc[:split_index], feats.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def fill_nan_values(X_train, X_test):
    col_to_fill_na = []
    X_train[col_to_fill_na] = X_train[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])
    X_test[col_to_fill_na] = X_test[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])
    return X_train, X_test

def create_folder_if_necessary(output_folderpath):
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()