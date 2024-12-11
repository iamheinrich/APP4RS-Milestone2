import rasterio
from safetensors.numpy import save
import lmdb
import os
from pathlib import Path
import pandas as pd
import tqdm
BAND_DICT = {
    'B01': None,
    'B02': None,
    'B03': None,
    'B04': None,
    'B05': None,
    'B06': None,
    'B07': None,
    'B08': None,
    'B09': None,
    'B10': None,
    'B11': None,
    'B12': None,
    'B8A': None
}


def get_band_safetensor_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        tmp_band_dict = {}
        for i in range(1, src.count + 1):
            content = src.read(i)
            tmp_band_dict[f'band_{i}'] = content
        st = save(tmp_band_dict)  # saves to a safetensors buffer
        return st


def convert_eurosat_dataset_to_lmdb(eurosat_dataset_path, lmdb_dir):
    """
    Convert the EuroSAT dataset to an LMDB database of safetensors.
    """
    lmdb_path = Path(lmdb_dir)
    lmdb_path.mkdir(parents=True, exist_ok=True)

    base_path = Path(eurosat_dataset_path)
    label_folders = [f for f in base_path.iterdir() if f.is_dir()]

    with lmdb.open(str(lmdb_path), map_size=int(1e12)) as env:
        with env.begin(write=True) as txn:
            for label_folder in tqdm.tqdm(label_folders, desc="Processing label folders", unit="folder"):
                # We assume only TIF files; let's assert this
                tif_files = list(label_folder.iterdir())
                for f in tif_files:
                    assert f.suffix.lower(
                    ) == '.tif', f"Non-TIF file found: {f}"

                # Store each tif file in LMDB
                for tif_file in tqdm.tqdm(tif_files, desc=f"Storing TIFFs for {label_folder.name}", unit="file", leave=False):
                    st = get_band_safetensor_from_tif(tif_file)
                    txn.put(tif_file.name.encode(), st)


def gather_metadata(input_data_path: str) -> pd.DataFrame:
    """
    Gather metadata for the EuroSAT dataset.
    For each class folder, assign 70% train, 15% validation, 15% test splits based on file index.
    """
    base_path = Path(input_data_path)
    label_folders = [f for f in base_path.iterdir() if f.is_dir()]

    all_entries = []
    for label_folder in tqdm.tqdm(label_folders, desc="Gathering metadata", unit="folder"):
        label = label_folder.name

        # Collect all tif files and assert no other files are present
        tif_files = list(label_folder.iterdir())
        for f in tif_files:
            assert f.suffix.lower() == '.tif', f"Non-TIF file found: {f}"

        n = len(tif_files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        for i, tif_path in enumerate(tif_files):
            stem = tif_path.stem
            current_label = stem.split("_")[0]
            # Determine split based on index i
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "validation"
            else:
                split = "test"

            all_entries.append({
                'id': tif_path.name,    # store filename as id
                'label': current_label,
                'split': split
            })

    df = pd.DataFrame(all_entries)
    return df


def store_metadata(df: pd.DataFrame, output_parquet_path: str):
    """
    Store metadata DataFrame as a parquet file.
    """
    df.to_parquet(output_parquet_path, index=False)


def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the EuroSAT dataset to lmdb and parquet format.

    :param input_data_path: path to the source EuroSAT dataset (root of the extracted zip file)
    :param output_lmdb_path: path to the destination EuroSAT lmdb file
    :param output_parquet_path: path to the destination EuroSAT parquet file
    :return: None
    """
    # Gather metadata
    metadata = gather_metadata(input_data_path)

    # Store metadata
    store_metadata(metadata, output_parquet_path)

    # Convert data to LMDB
    convert_eurosat_dataset_to_lmdb(input_data_path, output_lmdb_path)

    # Print the number of samples in the dataset and the number of samples in each split.
    num_keys = len(metadata)
    num_train_samples = len(metadata[metadata['split'] == 'train'])
    num_validation_samples = len(metadata[metadata['split'] == 'validation'])
    num_test_samples = len(metadata[metadata['split'] == 'test'])

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}")


if __name__ == "__main__":
    input_data_path = 'untracked-files/EuroSAT_MS'
    output_lmdb_path = 'untracked-files/datasets/EuroSAT'
    output_parquet_path = 'untracked-files/datasets/EuroSAT/metadata.parquet'
    main(input_data_path, output_lmdb_path, output_parquet_path)
