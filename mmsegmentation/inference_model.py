from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from pycocotools.coco import COCO
import os
import sys
import numpy as np
import pandas as pd
import albumentations as A
from torchvision.transforms import Resize
from torch import Tensor

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

root = "../../data/"

epoch = "20000"
cfg.work_dir = "./work_dirs/fcn_r50-d8"
cfg.gpu_ids = [1]
cfg.data.test.test_mode = True
cfg.data.test.pipeline[1]["img_scale"] = (512, 512) 

# cfg.data.samples_per_gpu = 4

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset, samples_per_gpu=cfg.data.samples_per_gpu , workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False
)
print(type(dataset))
print(type(data_loader))

# model 생성
checkpoint_path = os.path.join(cfg.work_dir, f"iter_{epoch}.pth")

model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")  # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader)
class_num = 10


prediction_strings = []
file_names = []
ann_file = root + 'test.json'
coco = COCO(ann_file)
img_ids = coco.getImgIds()
size = 256
transform = A.Compose([A.Resize(size, size)])
image = np.random.randint(10, size=512*512).reshape(512, 512)/255

for i, out in enumerate(output):
    prediction_string = ""
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    transformed = transform(image = image, mask = out)
    out = transformed['mask']
    for o in out:
        for j in o:
            prediction_string += (str(j) + " ")

            
    # print(len(prediction_string))
    prediction_strings.append(prediction_string)
    file_names.append(image_info["file_name"])
    
# print(cnt)
submission = pd.DataFrame()
submission["image_id"] = file_names
submission["PredictionString"] = prediction_strings
submission.to_csv(os.path.join(cfg.work_dir, f"test.csv"), index=None)
    
    
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# sys.stdout = open('stdout.txt', 'w')

# output

# sys.stdout.close()

# for문 활용해서 출력해보기