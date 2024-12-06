import os
import pandas as pd
import numpy as np
from PIL import Image
import lmdb
import tqdm
from safetensors.numpy import save

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


def convert_tif_file_to_numpy_array(tif_file_path):
    # Convert a .tif file to a numpy array
    with Image.open(tif_file_path) as img:
        return np.array(img)


def create_band_dictionary():
    # Create a dict with keys from the expected_resolutions, initialized as empty arrays
    band_dict = {band: np.array([]) for band in expected_resolutions.keys()}
    assert len(band_dict) == 12, "We expect each patch to have 12 bands"
    return band_dict


def convert_ben_dataset_to_lmdb(ben_dataset_path, lmdb_dir):
    """
    Convert the BEN dataset to an LMDB database of safetensor patches.

    :param ben_dataset_path: Path to the BEN dataset.
    :param lmdb_dir: Directory where the LMDB database will be stored.
    """
    # Ensure the LMDB directory exists
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)

    # First, collect all patch directories that contain at least one .tif file
    patch_dirs = []
    for root, dirs, files in os.walk(ben_dataset_path):
        if any(f.endswith('.tif') for f in files):
            patch_dirs.append(root)

    total_folders = len(patch_dirs)

    # Open the LMDB environment with a specified map size
    with lmdb.open(lmdb_dir, map_size=int(1e12)) as env:

        with env.begin(write=True) as txn:

            # Iterate over all patch directories with tqdm showing progress
            for root in tqdm.tqdm(patch_dirs, total=total_folders, desc="Processing folders", unit="folder"):
                patch_path = root
                band_dict = create_band_dictionary()

                tif_files = [f for f in os.listdir(
                    patch_path) if f.endswith('.tif')]

                for tif_file in tif_files:
                    # Extract band name (e.g., "B02") from filename
                    band_name = tif_file.split('_')[-1].replace('.tif', '')
                    arr = convert_tif_file_to_numpy_array(
                        os.path.join(patch_path, tif_file))

                    # Assert that the band's resolution matches the expected resolution
                    assert band_name in expected_resolutions, f"Unexpected band name: {
                        band_name}"
                    expected_res = expected_resolutions[band_name]
                    assert arr.shape[0] == expected_res, (
                        f"Resolution mismatch for band {band_name}: expected {
                            expected_res}, got {arr.shape[0]}"
                    )

                    band_dict[band_name] = arr

                # Convert the band dictionary to safetensor bytes
                tensor_bytes = save(band_dict)

                # Use the folder name as the key
                folder_name = os.path.basename(patch_path)
                txn.put(folder_name.encode(), tensor_bytes)


def count_samples_in_lmdb(lmdb_path: str) -> int:
    """Count the number of samples in the LMDB database."""
    with lmdb.open(lmdb_path, readonly=True) as env:
        with env.begin() as txn:
            return txn.stat()['entries']


def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the BigEarthNet dataset to lmdb and parquet format.

    :param input_data_path: path to the source BigEarthNet dataset (root of the tar file)
    :param output_lmdb_path: path to the destination BigEarthNet lmdb directory
    :param output_parquet_path: path to the destination BigEarthNet parquet file
    :return: None
    """

    # Convert the dataset to LMDB
    convert_ben_dataset_to_lmdb(input_data_path, output_lmdb_path)

    # Count total samples
    num_keys = count_samples_in_lmdb(output_lmdb_path)

    # Read metadata
    metadata_df = pd.read_parquet(
        'untracked-files/BigEarthNet-Lithuania-Summer/lithuania_summer.parquet')

    # Store as parquet file in output_parquet_path
    metadata_df.to_parquet(output_parquet_path)

    # Count the number of samples in the dataset and the number of samples in each split.
    num_train_samples = len(metadata_df[metadata_df['split'] == 'train'])
    num_validation_samples = len(
        metadata_df[metadata_df['split'] == 'validation'])
    num_test_samples = len(metadata_df[metadata_df['split'] == 'test'])

    assert num_keys == num_train_samples + num_validation_samples + num_test_samples, (
        f"The number of keys in the LMDB database is {
            num_keys} which is not equal to "
        f"the number of samples in the dataset {
            num_train_samples + num_validation_samples + num_test_samples}"
    )

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}")


if __name__ == "__main__":
    main(
        input_data_path='untracked-files/BigEarthNet-Lithuania-Summer',
        output_lmdb_path='untracked-files/datasets',
        output_parquet_path='untracked-files/datasets/metadata/metadata_ben.parquet'
    )
