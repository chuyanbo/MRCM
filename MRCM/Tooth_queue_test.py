import torch
from torch import nn , optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from Tooth_2Ddataloader import Tooth , CBCT_path ,DH_img_path ,DH_label_path,all_data_path,all_label_path,all_queue_data_path,all_queue_label_path
from Tooth_2Dnet import Resnet18 , UNet , config
from Tooth_2DLOSS import DiceLoss0,WeightedFocalLoss , FocalLoss , Multi_FocalLoss
from uct_nets.UCTransNet import UCTransNet
from uct_nets.UNet import UNetc
from UNet2D_BraTs.unet import UnetB
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from PIL import Image

multiGPU = True
if multiGPU :
    import os
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda")


batchsz2=1
lr=1e-2
epochs=5
classes=6
path_SRT="/mnt"

para_save=path_SRT+"/LLSRT/para/q_resnet1.1.mdl"
para_load=path_SRT+"/LLSRT/para/q_resnet1.1.mdl"
para_save_u3=path_SRT+"/LLSRT/para/q_Unet_focal1.1.pth"
para_load_u3=path_SRT+"/LLSRT/para/q_Unet_focal1.1.pth"

# 1.1 class=6 ratio = (1,1)
# 1.2 class=7 ratio = (1,1) （abandon）
# 1.3 class=6 ratio = (1,1.5)
config3=config(1,1)
model4 = UCTransNet(config3,n_channels=1,n_classes=classes,img_size=256)
model4.to(device)
if multiGPU :
    model4 = torch.nn.parallel.DistributedDataParallel(model4, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

#torch.manual_seed(8848)
print(sum(p.numel() for p in model4.parameters()))
torch.cuda.empty_cache()

tooth_DH_db_eval=Tooth(all_queue_data_path,all_queue_label_path,(256,256,256),mode='eval',load_mode="all",num_class=classes)
tooth_DH_loader_eval = DataLoader(tooth_DH_db_eval,batch_size=batchsz2,shuffle=False,num_workers=2,pin_memory=True,
                                  sampler=DistributedSampler(tooth_DH_db_eval,shuffle=False))

def Unet2_main():
    # model3.load_state_dict(torch.load(para_load_u2))
    
    model4.load_state_dict(torch.load(para_load_u3))
    torch.cuda.empty_cache()
    
    if torch.cuda.device_count() > 1:
        print("we can use ", torch.cuda.device_count(), "GPUs.")
        
    print("__epoch start test__")
    model4.eval()
    with torch.no_grad():
        step=0
        color = tooth_DH_db_eval.color
        for idx,(x,y,label) in enumerate(tooth_DH_loader_eval):
            x , y , label = x.to(device) ,  y.to(device) , label.to(device)
            x0=model4(x)
            mask0_value = torch.max(F.softmax(x0,dim=1),dim=1)[0].data.unsqueeze(1)
            mask0 = torch.max(F.softmax(x0,dim=1),dim=1)[1].unsqueeze(1)
            #print(mask0_value.sum())
            #mask0 = torch.ones_like(y)*(epoch+1)
            color_id = 0
            color_lim = 0.4
            for cc in color:
                mask_c1 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[0],0)
                mask_c2 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[1],0)
                mask_c3 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[2],0)
                if color_id == 0 :
                    mask = torch.cat((mask_c1,mask_c2,mask_c3),dim=1)       
                else:
                    mask += torch.cat((mask_c1,mask_c2,mask_c3),dim=1)                
                color_id += 1
            
            im = Image.fromarray((mask[0].cpu()).transpose(0,-1).transpose(0,1).numpy().reshape(256,256,-1).astype(np.uint8)).convert("RGB")
            im.save("/mnt/LLSRT/queue_result/test_queue_"+str(idx//100)+str((idx//10)%10)+str(idx%10)+"_.jpeg")
                
    print("__test finish and saved__")
        
        

            

if __name__=='__main__':
    Unet2_main()
    
"""
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 /mnt/LLSRT/Tooth_queue_test.py
"""