import os
from sklearn.datasets import get_data_home

# BBScore dataset class
from data.SSV2_pruned import SSV2PrunedStimulusTrainSet
from data.SSV2_pruned import SSV2PrunedStimulusTestSet


def main():

    # BBScore stores datasets in SCIKIT_LEARN_DATA
    data_root = os.environ.get("SCIKIT_LEARN_DATA", get_data_home())

    print(f"Dataset storage directory: {data_root}")

    print("\nDownloading SSV2 training set...\n")

    train_dataset = SSV2PrunedStimulusTrainSet(
        root_dir=os.path.join(data_root, "SSV2Pruned"),
        overwrite=False,
        preprocess=lambda x: x   # identity function just to allow loading
    )

    print(f"Training dataset loaded with {len(train_dataset)} videos")

    print("\nDownloading SSV2 test set...\n")

    test_dataset = SSV2PrunedStimulusTestSet(
        root_dir=os.path.join(data_root, "SSV2Pruned"),
        overwrite=False,
        preprocess=lambda x: x
    )

    print(f"Test dataset loaded with {len(test_dataset)} videos")

    print("\nSSV2 dataset download complete.")


if __name__ == "__main__":
    main()