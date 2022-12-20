import fiftyone as fo
import json

# from .mmsegmentation.mmseg.datasets.boost import BOOSTDataset

name = "my-dataset"
data_path = "../data/mmseg/images/train/"
labels_path = "../data/mmseg/annotations/train/"

# dataset_type = BOOSTDataset

# Create the dataset
dataset = fo.Dataset.from_dir(
    data_path=data_path,
    labels_path=labels_path,
    dataset_type=fo.types.ImageSegmentationDirectory,
    # dataset_type=dataset_type,
    name=name,
)


# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

session = fo.launch_app(dataset)
session.wait()
