_base_ = [
    "./models/fcn_r50-d8.py",
    "./datasets/coco.py",
    "./default_runtime.py",
    "./schedules/schedule_20k.py",
]
model = dict(decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
