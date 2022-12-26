import os
import warnings 
warnings.filterwarnings('ignore')

import torch
from utils import label_accuracy_score, add_hist

import numpy as np
from tqdm import tqdm

import albumentations as A

import wandb


print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"

category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(f"[Epoch {epoch} Validation]")
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                pbar.update(1)
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch} \nAverage Loss: {round(avrg_loss.item(), 4)}, \tAccuracy : {round(acc, 4)}, \tmIoU: {round(mIoU, 4)}')
        print(f'IoU by class : \n{IoU_by_class}')

        for d in IoU_by_class:
            cls = list(d.keys())[0]
            iou = list(d.values())[0]
            wandb.log({
                f"Valid_IoU_By_Class/{cls}" : iou
            })

        wandb.log({
                "Valid/loss": loss,
                "Valid/mIOU": mIoU,
                "Valid/acc": acc,
                "Valid/acc_cls": acc_cls
                })
        
    return avrg_loss, mIoU

def save_model(model, saved_dir, file_name='SMP_PAN_SwinL.pt'):    ########## 저장 파일 이름 바꿔주세요 ##########
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = 0
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(f"[Epoch {epoch+1} Train]")
            for step, (images, masks, _) in enumerate(data_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                
                # device 할당
                model = model.to(device)
                
                # inference
                outputs = model(images)
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                pbar.update(1)
                val_dict = {
                    "loss": round(loss.item(),4),
                    "mIoU": round(mIoU,4),
                    "lr": optimizer.param_groups[0]['lr']
                }
                pbar.set_postfix(val_dict)

                # step 주기에 따른 loss 출력
                # if (step + 1) % 25 == 0:
                #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], \
                #             Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

                wandb.log({
                    "Train/loss": loss,
                    "Train/mIOU": mIoU,
                    "Train/acc": acc,
                    "Train/acc_cls": acc_cls,
                    "LearningRate" : optimizer.param_groups[0]['lr']
                })
            scheduler.step()
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            if best_mIoU < val_mIoU:
                print(f">>>>>>>>>> Best performance at epoch: {epoch + 1} <<<<<<<<<<")
                best_epoch = epoch + 1
                print(f">>>>>>>>>> Save model in {saved_dir} <<<<<<<<<<")
                best_mIoU = val_mIoU
                save_model(model, saved_dir)
            
            print(f">>>>>>>>>> BEST EPOCH : {best_epoch} <<<<<<<<<<")
            print(f">>>>>>>>>> BEST mIoU : {best_mIoU} <<<<<<<<<<")

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array