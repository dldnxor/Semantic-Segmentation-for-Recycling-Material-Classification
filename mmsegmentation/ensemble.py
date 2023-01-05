import pandas as pd
import numpy as np
import os

work_dir = './files'
sub1= work_dir + '/hyunsoo.csv'
sub2= work_dir + '/juhee.csv'
sub3= work_dir + '/haewook.csv'
sub1_miou = 0.6989 # SMP_PAN_SwinL_StepLR_AUGMANY300_best_model(pretrained).
sub2_miou = 0.6961 # upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_all
sub3_miou = 0.6798 # SMP_PAN_SwinL_StepLR


sub1_df = pd.read_csv(sub1)
sub2_df = pd.read_csv(sub2)
sub3_df = pd.read_csv(sub3)

# print(sub1_df['image_id'])

file_names = []
prediction_string = ''
prediction_strings = []
test = ''
break_point = len(sub1_df['PredictionString'][0])

standard = sub1_df['PredictionString'][0].split()
print(len(standard))
# print(len(sub1_df['image_id']))


for i in range(len(sub1_df['image_id'])):
    test1 = sub1_df['PredictionString'][i].split()
    test2 = sub2_df['PredictionString'][i].split()
    test3 = sub3_df['PredictionString'][i].split()
        
    for j in range(len(standard)):
        
        test_set={'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0}
        # print(test1[j])
        str1 = test1[j]
        str2 = test2[j]
        str3 = test3[j]
        
        test_set.update({str1: test_set[str1] + sub1_miou})
        test_set.update({str2: test_set[str2] + sub2_miou}) 
        test_set.update({str3: test_set[str3] + sub3_miou})
                    
        selected = max(test_set, key=test_set.get)
        
        prediction_string += (selected+' ')
            
    prediction_strings.append(prediction_string)
    prediction_string = ''
        

submission = pd.DataFrame()
submission["image_id"] = sub1_df['image_id']
submission["PredictionString"] = prediction_strings
submission.to_csv(os.path.join(work_dir, f"ensemble.csv"), index=None)
