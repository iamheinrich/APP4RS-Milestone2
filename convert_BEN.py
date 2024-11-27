# imports


# expected resolutions for the BigEarthNet dataset
expected_resolutions = {
    'B01': 20,
    'B02': 120,
    'B03': 120,
    'B04': 120,
    'B05': 60,
    'B06': 60,
    'B07': 60,
    'B08': 120,
    'B8A': 60,
    'B09': 20,
    'B11': 60,
    'B12': 60
}


def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the BigEarthNet dataset to lmdb and parquet format.

    :param input_data_path: path to the source BigEarthNet dataset (root of the tar file)
    :param output_lmdb_path: path to the destination BigEarthNet lmdb file
    :param output_parquet_path: path to the destination BigEarthNet parquet file
    :return: None
    """

    # TODO: Read the tif files and write them to the lmdb file.
    # TODO: Write the metadata to a parquet file.
    # TODO: Print the number of samples in the dataset and the number of samples in each split.

    num_keys = ...
    num_train_samples = ...
    num_validation_samples = ...
    num_test_samples = ...

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}")
