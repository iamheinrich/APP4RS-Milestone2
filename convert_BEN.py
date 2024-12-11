import pandas as pd
import numpy as np
from PIL import Image
import lmdb
import tqdm
from safetensors.numpy import save
from pathlib import Path

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


def convert_tif_file_to_numpy_array(tif_file_path: Path):
    # Convert a .tif file to a numpy array
    with Image.open(tif_file_path) as img:
        return np.array(img)


def create_band_dictionary():
    # Create a dict with keys from the expected_resolutions, initialized as empty arrays
    band_dict = {band: np.array([]) for band in expected_resolutions.keys()}
    assert len(band_dict) == 12, "We expect each patch to have 12 bands"
    return band_dict


def convert_ben_dataset_to_lmdb(ben_dataset_path: str, lmdb_dir: str):
    """
    Convert the BEN dataset to an LMDB database of safetensor patches.
    """
    ben_dataset_path = Path(ben_dataset_path)
    lmdb_dir = Path(lmdb_dir)

    # Ensure the LMDB directory exists
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Find all patch directories that contain at least one .tif file
    patch_dirs = []
    for dir_path in ben_dataset_path.glob('**/'):
        if dir_path.is_dir():
            tif_files = list(dir_path.glob('*.tif'))
            if tif_files:
                patch_dirs.append(dir_path)

    total_folders = len(patch_dirs)

    # Open the LMDB environment with a specified map size
    # Map size is set to 3GB for BEN, as previously discussed
    with lmdb.open(str(lmdb_dir), map_size=int(3e9)) as env:
        with env.begin(write=True) as txn:
            # Iterate over all patch directories with tqdm showing progress
            for patch_path in tqdm.tqdm(patch_dirs, total=total_folders, desc="Processing folders", unit="folder"):
                band_dict = create_band_dictionary()
                tif_files = list(patch_path.glob('*.tif'))

                # Since each folder only contains 12 images, no need for a tqdm here
                for tif_file in tif_files:
                    # Extract band name (e.g., "B02") from filename
                    band_name = tif_file.stem.split('_')[-1]
                    arr = convert_tif_file_to_numpy_array(tif_file)

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
                folder_name = patch_path.name
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

    # Remove the country column as discussed
    if 'country' in metadata_df.columns:
        metadata_df = metadata_df.drop(columns=['country'])

    # Store as parquet file in output_parquet_path
    metadata_df.to_parquet(output_parquet_path)

    # Count the number of samples in each split.
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
    input_data_path = 'untracked-files/BigEarthNet-Lithuania-Summer'
    output_lmdb_path = 'untracked-files/datasets/BigEarthNet'
    output_parquet_path = 'untracked-files/datasets/BigEarthNet/metadata.parquet'
    main(input_data_path, output_lmdb_path, output_parquet_path)
