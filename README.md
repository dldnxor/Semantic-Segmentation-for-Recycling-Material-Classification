# 🏆재활용 품목 분류를 위한 Semantic Segmentation 🏆
![](https://velog.velcdn.com/images/tls991105/post/0d741b81-29d7-4582-bcd7-eed3370d8049/image.png)
***
## 🔥Member
| [정승윤](https://github.com/syoon6682) | [김주희](https://github.com/alias26) | [신현수](https://github.com/Hyun-soo-Shin) | [이우택](https://github.com/dldnxor) | [이해욱](https://github.com/woooo-k) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/syoon6682" width="100"> | <img src="https://avatars.githubusercontent.com/alias26" width="100"> | <img src="https://avatars.githubusercontent.com/Hyun-soo-Shin" width="100"> | <img src="https://avatars.githubusercontent.com/dldnxor" width="100"> | <img src="https://avatars.githubusercontent.com/woooo-k" width="100"> |
***
## Index
- [🏆재활용 품목 분류를 위한 Semantic Segmentation 🏆](#재활용-품목-분류를-위한-semantic-segmentation-)
  - [🔥Member](#member)
  - [Index](#index)
  - [🏅Project Summary](#project-summary)
  - [👨‍👩‍👧‍👧Team Roles](#team-roles)
  - [🗃️Procedures](#️procedures)
  - [🌿Features](#features)
  - [📊Result](#result)
    - [탐색적 분석(EDA) 및 데이터 전처리](#탐색적-분석eda-및-데이터-전처리)
    - [모델 개요](#모델-개요)
    - [Data Augmentation](#data-augmentation)
    - [Ensemble](#ensemble)
    - [시연결과](#시연결과)
  - [👨‍💻Conclusion](#conclusion)
  - [💻Requirements](#requirements)
  - [🏗️Folder Structure](#️folder-structure)
***
## 🏅Project Summary

>### - 프로젝트 주제
> 쓰레기 이미지에서 Semantic Segmentation을 활용하여 10종류 클래스의 쓰레기로 추측할 수 있다. 카메라를 이용한 분리수거 판별을 통해 올바르게 분리수거가 되어 있는지 알 수 있다.
>
>### - 개요 및 기대효과
>Semantic Segmentation Task 대회를 진행하면서 EDA, Modeling, Ensemble 등 다양한 테스크를 경험해볼 수 있고 이를 수행하면서 Semantic Segmentation에 대한 이해도를 높일 수 있다. 기대 효과로는 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 활용
>
>### - 활용 장비 및 재료 
> 서버: V100 GPU  
> 라이브러리: MMSegmentation  
> 개발 및 협업 툴: Git, Slack, Zoom, Visual Studio Code
>
> ### - 데이터 셋의 구조도
> - **데이터셋 통계**
>- 전체 이미지 개수 : 3272장
>- 11 class : Background ,General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
>- 이미지 크기 : (512, 512)
>- 데이터셋 형태 : COCO Dataset
>> - images :
>>    - id : 파일 안에서 image 고유 id, ex) 1
>>    - height : 512
>>    - width : 512
>>    - filename : ex) batch01_vt/002.jpg
>
---
## 👨‍👩‍👧‍👧Team Roles
>- **이우택**: SMP에서 모델 실험 진행, 다양한 augmentation 기법 활용
>- **정승윤**: MMsegmentation segformer 학습, inference file 구현, ensemble file 구현 
>- **김주희**: Wandb Setting, Fiftyone, transfer dataset for MMSeg, Albumentation 추가, UperNet Swin_Large/Tiny 모델 학습
>- **이해욱**: SMP에서 모델 실험 진행, Swin-L encoder 추가, AMP(Mixed-Precision) 추가, UNet ResNet101 / PAN Swin_Large 모델 학습
>- **신현수**: EDA, Wandb Setting, SMP에서 모델 실험 진행, DiceFocalLoss, CDB Loss 구현 및 학습
---
## 🗃️Procedures
>**[2022.12.19 ~ 2022.12.23]**
>- Baseline 코드 setting
>   - 공통
>     - Fiftyone 시각화
>   - mmseg
>     - MMsegmentation install 및 환경 설정 
>     - inference file 구현
>   - Baseline Code
>      - SMP 환경 설정
>      - inference file 구현
>   - 각자 모델 선정
>     - 주희: MMSeg, UperNet Swin_Large/Tiny 
>     - 승윤: MMSeg, Segformer B5
>     - 해욱: SMP, UNet ResNet101 / PAN Swin_Large
>     - 우택: SMP, PAN Swin_Large
>     - 현수: SMP, PAN Swin_Large
>     
><br>
>
>**[2022.12.24 ~ 2022.12.27]**
>- Augmentation 실험
>   - 각자 선정한 모델들을 선정하여 각 모델에 맞는 Augmentation 실험 진행 
><br>
>
>**[2022.12.28 ~ 2023.01.01]**
>- Refresh Day
>
>**[2023.01.02 ~ 2023.01.04]**
>- Augmentation을 토대로 선정한 모델 학습
>- Model ensemble file 구현
>
>**[2023.01.05]**
>- 모델 ensemble 및 대회 마무리
>- Git merge 및 Readme 작성
>- Wrap-Up Report 작성


---
## 🌿Features
>**feat/fiftyone**: FiftyOne을 통한 데이터시각화  <br/>
>**feat/upernet**: UperNet Swin 모델 실험   <br/>
>**feat/segformer**: Segformer 실험 및 ensemble 파일 작성 <br/>
>**feat/inferencer**: MMsegmentation inference 파일 작성 <br/>
>**feat-losses**: Custom Criterion 추가<br/>
>**feat-baseline_setting**: baseline update<br/>
>**feat-SMP**: Baseline Code에 SMP 라이브러리 모델 추가<br/>
>**feat/SMP-AMP**: SMP Code에 AMP(Mixed-Precision) 추가<br/>
>**feat/SMP-swin**: SMP Code에 Swin 모델 추가<br/>
>
---
## 📊Result
### 탐색적 분석(EDA) 및 데이터 전처리
>> * **Class 당 Annotation 분포**
>>
>> <img src="https://velog.velcdn.com/images/tls991105/post/cbd10893-8e75-48b5-bf6c-5b8f0251b7bd/image.png" width="600"/>
>
>
>>* **이미지 당 Annotation 수**
>>
>><img src="https://velog.velcdn.com/images/tls991105/post/12ebc0ef-e4c8-4162-9fd2-84013287948b/image.png" width="600"/>
>
>
>>* **하나의 이미지가 가지는 class 수 분포**
>>
>><img src="https://velog.velcdn.com/images/tls991105/post/313ee403-ff6d-4f23-b5b5-cf3c851657fb/image.png" width="600"/>
>
>
>>* **Annotation의 크기 분포**
>>
>><img src="https://velog.velcdn.com/images/tls991105/post/3c67ce48-802f-4742-9915-7ad8bf1158af/image.png" width="600"/>
>
>
>>* **Class별 Annotation의 크기 분포**
>>
>><img src="https://velog.velcdn.com/images/tls991105/post/05058ec6-2605-467f-8ac6-14161b72d5c7/image.png" width="600"/> 
>
>

---

### 모델 개요
>* **사용한 모델**<br/>
>  SMP : Swin-L + PAN + ImageNet(pretrained)<br/>
>  MSeg : Upernet Swin-L(Pretrained)
>  
>* **Metric** : MIoU
>
>||SMP|MMSeg|
>|----|----|----|
>|Loss|Cross Entropty|Cross Entropy|
>|Optimizer|AdamW|SGD|
>|Schedulers|CosineAnneling|IterBased|

### Data Augmentation
> Base: UperNet Swin_Tiny<br/>
> Iteration: 160000<br/>
>|Augmentation|MIoU|
>|----|----|
>|Base|0.5837|
>|+RandMosaic|0.5908|
>|+CutOut|0.5953|
>|+Rotate|0.5952|
>|+GaussNoise|0.5912|
>|+Mblur|0.6138|
> 
>**Result**    
>Model: UperNet Swin_Large 
>Iteration: 160000 
>MIoU: 0.6961

>Base: Segformer_B0 <br/>
>Iteration: 120000 <br/>
>|Augmentation|MIoU|
>|----|----|
>|Base|0.5837|
>|+Optical distortion|0.5958|
>|+Mosaic|0.5967|
>|+Blur|0.6033|
>|+Flip|0.6058|
> 
>**Result**  
>Model: Segformer B5 
>Iteration: 160000 
>MIoU: 0.6526

> Base: PAN Swin Large<br/>
>|Augmentation|MIoU|
>|----|----|
>|Base|0.6798|
>|+Flip +RandomSizedCrop|0.6754|
>|+MaskDropOut|0.6980|
>|+RandomBrightness Contrast|0.6697|
>|+GaussNoise|0.5912|
>|+Mblur|0.6138|


### Ensemble
>모델 별 mIoU를 기반으로 한 가중치를 부여하여 픽셀별 Hard voting을 통한 앙상블
>| Model1| Model2 | Model3 || **Ensemble** |
>|:---:|:---:|:---:|-|:---:|
>| 0.7234 | 0.7143 | 0.6961 || **0.7342** |
>
>Model1 : (SMP) PAN Swin Large 1  
>Model2 : (SMP) PAN Swin Large 2  
>Model3 : (MMSeg) UperNet Swin Large


### 시연결과
>지금까지 학습한 모델을 최종 앙상블해서 제출
>|MIoU|
>|----|
>|0.7264|
---
## 👨‍💻Conclusion
>#### 잘한 점들
>1. baseline model setting이 매우 빨랐고 이를 통해 빠른 실험 시작이 가능했다
>2. augmentation을 위해 Albumentation에만 국한된 것이 아닌, 다른 library를 탐색하고 공부해보는 시간을 가졌다
>3. SMP와 MMSegmentation 두 library를 이용해 다양한 실험을 할 수 있었다. 
>
>#### 아쉬운 점들
>1. 데이터 전처리를 해보지 못했다.
>2. DenseCRF, TTA, model soup과 같은 결과에 도움이 될만한 기법들을 시도해 보았으나 큰 효과를 얻지 못하였다.
>3. Optuna 등의 library를 사용한 Hyperparameter tuning을 시도해보지 못했다.
>4. Copy-Paste augmentation을 적용하지 못했다.
>
> #### 프로젝트를 통해 배운점
>1. Semantic Segmentation에 대한 이론 및 모델 학습
>2. 다양한 Loss Function, Optimizer, Scheduler에 대한 이해 및 실험 

---
## 💻Requirements
MMSeg
```
conda install pytorch=1.7.1 cudatoolkit=11.0 torchvision -c pytorch  
pip install openmim  
mim install mmseg  
```
SMP
```
pip install -r requirements.txt
pip install segmentation-models-pytorch
```
To use fiftyone
```
pip install fiftyone
mkdir fiftyone
chmod ... fiftyone
export FIFTYONE_DATABASE_DIR fiftyone
```

---
## 🏗️Folder Structure
```
├── code (SMP)
│    ├── 📂models
│    │      ├── 📝build.py
│    │      └── 📝swin.py
│    ├── 📂submission
│    ├── 📝SMP_dataset.py
│    ├── 📝SMP_train.py
│    ├── 📝SMP_inference.py
│    ├── 📝custom_scheduler.py
│    ├── 📝losses.py
│    └── ...
│   
└── mmsegmentation 
     ├── 📂segformer_trash
     ├── 📂upernet_swin_trash  
     └── ... 
```
---
