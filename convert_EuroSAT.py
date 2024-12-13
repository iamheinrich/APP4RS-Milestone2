import rasterio
from safetensors.numpy import save
import lmdb
import os
import pandas as pd

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
        st = save(tmp_band_dict)  # Saves to a safetensors buffer
        return st

def convert_eurosat_dataset_to_lmdb(eurosat_dataset_path, lmdb_dir):
    """
    Convert the EuroSAT dataset to an LMDB database of safetensors.
    """
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)

    label_folders = [
        os.path.join(eurosat_dataset_path, d) for d in os.listdir(eurosat_dataset_path)
        if os.path.isdir(os.path.join(eurosat_dataset_path, d))
    ]

    with lmdb.open(lmdb_dir, map_size=int(1e12)) as env:
        with env.begin(write=True) as txn:
            for label_folder in label_folders:
                tif_files = [
                    os.path.join(label_folder, f) for f in os.listdir(label_folder)
                    if f.endswith('.tif')
                ]

                for tif_file in tif_files:
                    st = get_band_safetensor_from_tif(tif_file)
                    txn.put(os.path.basename(tif_file).encode(), st)


def gather_metadata(input_data_path: str) -> pd.DataFrame:
    """
    Gather metadata for the EuroSAT dataset.
    For each class folder, assign 70% train, 15% validation, 15% test splits based on file index.
    """
    label_folders = [
        os.path.join(input_data_path, d) for d in os.listdir(input_data_path)
        if os.path.isdir(os.path.join(input_data_path, d))
    ]

    all_entries = []
    for label_folder in label_folders:
        label = os.path.basename(label_folder)

        tif_files = [
            os.path.join(label_folder, f) for f in os.listdir(label_folder)
            if f.endswith('.tif')
        ]

        n = len(tif_files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        for i, tif_path in enumerate(tif_files):
            stem = os.path.splitext(os.path.basename(tif_path))[0]
            current_label = stem.split("_")[0]

            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "validation"
            else:
                split = "test"

            all_entries.append({
                'class_name': current_label,
                'patch_name': os.path.basename(tif_path).removesuffix(".tif"),
                'split': split
            })

    df = pd.DataFrame(all_entries)
    return df

def store_metadata(df, output_parquet_path):
    """
    Store metadata DataFrame as a parquet file.
    """
    os.makedirs(
        os.path.dirname(output_parquet_path),
        exist_ok=True
    )
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
