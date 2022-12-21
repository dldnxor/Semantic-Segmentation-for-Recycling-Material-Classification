from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
import os
import sys
import numpy as np

classes = (
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
)


cfg = Config.fromfile("./trash/models/fcn_r50-d8.py")

root = "../../data/mmseg/test"

epoch = "20000"
cfg.work_dir = "./work_dirs/fcn_r50-d8"
cfg.gpu_ids = [1]
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
)

# model 생성
checkpoint_path = os.path.join(cfg.work_dir, f"iter_{epoch}.pth")

model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")  # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader)


# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# sys.stdout = open('stdout.txt', 'w')

print(output)

# sys.stdout.close()

# for문 활용해서 출력해보기