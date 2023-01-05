# dataset settings 자신의 맞는 경로 수정하기
dataset_type = "CustomDataset"
data_root = "../../data/mmseg/"

# class settings
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
palette = [
    [0, 0, 0],
    [192, 0, 128],
    [0, 128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192],
]

# set normalize value
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    # dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations"),
    dict(type="RandomMosaic", img_scale=(512, 512), center_ratio_range=(0.8, 1.1), prob=0.3),
    dict(
        type="RandomCutOut",
        n_holes=(3, 5),
        cutout_ratio=[(0.1, 0.1), (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)],
        seg_fill_in=1,
        prob=0.4,
    ),
    dict(type="Albumentations", transform="MaskDropout", max_objects=2, mask_fill_value=0, p=0.5),
    dict(type="Resize", img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomRotate", degree=45, prob=0.4),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Albumentations", transform="GaussNoise"),
    dict(type="Albumentations", transform="MotionBlur"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="MultiImageMixDataset",
        dataset=[
            dict(
                classes=classes,
                palette=palette,
                type=dataset_type,
                reduce_zero_label=False,
                img_dir=data_root + "images/train",
                ann_dir=data_root + "annotations/train",
                pipeline=[
                    dict(type="LoadImageFromFile"),
                    dict(type="LoadAnnotations"),
                ],
            )
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/val",
        ann_dir=data_root + "annotations/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "test",
        pipeline=test_pipeline,
    ),
)
