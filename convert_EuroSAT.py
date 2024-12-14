import rasterio
from safetensors.numpy import save
import lmdb
import os
import pandas as pd

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
    os.makedirs(
        os.path.dirname(lmdb_dir), # .dirname turns untracked-files/datasets/EuroSAT.lmdb -> untracked-files/datasets, no problem because lmdb creats .lmdb dir
        exist_ok=True
    )

    label_folder_paths = [
        os.path.join(eurosat_dataset_path, d) for d in os.listdir(eurosat_dataset_path)
        if os.path.isdir(os.path.join(eurosat_dataset_path, d))
    ]

    with lmdb.open(lmdb_dir, map_size=int(1e12)) as env:
        with env.begin(write=True) as txn:
            for label_folder_path in label_folder_paths:
                tif_file_paths = [
                    os.path.join(label_folder_path, f) for f in os.listdir(label_folder_path)
                    if f.endswith('.tif')
                ]

                for tif_file_path in tif_file_paths:
                    st = get_band_safetensor_from_tif(tif_file_path)
                    txn.put(os.path.basename(tif_file_path).removesuffix(".tif").encode(), st)


def gather_metadata(input_data_path: str) -> pd.DataFrame:
    """
    Gather metadata for the EuroSAT dataset.
    For each class folder, assign 70% train, 15% validation, 15% test splits based on file index.
    """
    label_folder_paths = [
        os.path.join(input_data_path, d) for d in os.listdir(input_data_path)
        if os.path.isdir(os.path.join(input_data_path, d))
    ]

    all_entries_metadata = []
    for label_folder_path in label_folder_paths:
        tif_paths = [
            os.path.join(label_folder_path, f) for f in os.listdir(label_folder_path)
            if f.endswith('.tif')
        ]

        n = len(tif_paths)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        # input_data_path -> label_folder_paths -> tif_paths
        for tif_path in tif_paths:
            stem = os.path.basename(tif_path).removesuffix(".tif")
            current_label = stem.split("_")[0]
            band_number = int(stem.split("_")[1])

            if band_number <= n_train:
                split = "train"
            elif band_number <= n_train + n_val:
                split = "validation"
            else:
                split = "test"

            all_entries_metadata.append({
                'class_name': current_label,
                'patch_name': os.path.basename(tif_path).removesuffix(".tif"),
                'split': split
            })

    metadata_df = pd.DataFrame(all_entries_metadata)
    return metadata_df

def store_metadata(df, output_parquet_path):
    """
    Store metadata DataFrame as a parquet file.
    """
    os.makedirs(
        os.path.dirname(output_parquet_path),
        exist_ok=True
    )
    df.to_parquet(output_parquet_path, index=False)

def count_samples_in_lmdb(lmdb_path):
    """Count the number of samples in the LMDB database."""
    with lmdb.open(lmdb_path, readonly=True) as env:
        with env.begin() as txn:
            return txn.stat()['entries']

def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the EuroSAT dataset to lmdb and parquet format.

    :param input_data_path: path to the source EuroSAT dataset (root of the extracted zip file)
    :param output_lmdb_path: path to the destination EuroSAT lmdb file
    :param output_parquet_path: path to the destination EuroSAT parquet file
    :return: None
    """
    metadata = gather_metadata(input_data_path)
    store_metadata(metadata, output_parquet_path)
    convert_eurosat_dataset_to_lmdb(input_data_path, output_lmdb_path)
    
    num_keys = count_samples_in_lmdb(output_lmdb_path)

    # Print the number of samples in the dataset and the number of samples in each split.
    num_train_samples = len(metadata[metadata['split'] == 'train'])
    num_validation_samples = len(metadata[metadata['split'] == 'validation'])
    num_test_samples = len(metadata[metadata['split'] == 'test'])

    assert num_keys == num_train_samples + num_validation_samples + num_test_samples, (
        f"The number of keys in the LMDB database is {num_keys} which is not equal to "
        f"the number of samples in the dataset {num_train_samples + num_validation_samples + num_test_samples}"
    )

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}")

if __name__ == "__main__":
    input_data_path = 'untracked-files/EuroSAT_MS'
    output_lmdb_path = 'untracked-files/datasets/EuroSAT.lmdb'
    output_parquet_path = 'untracked-files/datasets/metadata.parquet'
    main(input_data_path, output_lmdb_path, output_parquet_path)
