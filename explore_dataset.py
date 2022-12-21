import fiftyone as fo
import json

train_name = "train"
val_name = "val"
data_path = "../data/"
train_labels_path = "../data/train.json"
val_labels_path = "../data/val.json"


train_dataset = fo.Dataset.from_dir(
    data_path=data_path,
    labels_path=train_labels_path,
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["segmentations"],
    name=train_name,
)

val_dataset = fo.Dataset.from_dir(
    data_path=data_path,
    labels_path=val_labels_path,
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["segmentations"],
    name=val_name,
)


if __name__ == "__main__":
    session = fo.launch_app(train_dataset)
    session.dataset = val_dataset
    session.wait()
