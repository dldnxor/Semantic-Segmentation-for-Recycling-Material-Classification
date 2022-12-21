import fiftyone as fo
import json

# from .mmsegmentation.mmseg.datasets.boost import BOOSTDataset

train_name = "train"
train_data_path = "../data/mmseg/images/train/"
train_labels_path = "../data/mmseg/annotations/train/"

val_name = "val"
val_data_path = "../data/mmseg/images/val/"
val_labels_path = "../data/mmseg/annotations/val/"

labels = {
    0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}

classes = [
    "Background",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

# Create the dataset
train_dataset = fo.Dataset.from_dir(
    data_path=train_data_path,
    labels_path=train_labels_path,
    dataset_type=fo.types.ImageSegmentationDirectory,
    compute_metadata=True,
    name=train_name,
)

val_dataset = fo.Dataset.from_dir(
    data_path=val_data_path,
    labels_path=val_labels_path,
    dataset_type=fo.types.ImageSegmentationDirectory,
    compute_metadata=True,
    name=val_name,
)

# create default segmentation label
train_dataset.default_mask_targets = labels
val_dataset.default_mask_targets = labels
train_dataset.save()
val_dataset.save()

# View summary info about the dataset
# print(train_dataset)
# print(val_dataset)

# # Print the first few samples in the dataset
# print(train_dataset.head())
# print(val_dataset.head())


if __name__ == "__main__":
    session = fo.launch_app(train_dataset)
    session.dataset = val_dataset
    session.wait()
