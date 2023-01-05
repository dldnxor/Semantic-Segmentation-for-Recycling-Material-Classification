# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="MMSegWandbHook",
            init_kwargs=dict(
                project="semantic_segmentation",
                entity="cv_09_semanticsegmentation",
                name="test",
            ),
            interval=2000,
            by_epoch=False,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=100,
        ),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ],
)

# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
