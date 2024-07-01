
import torch
from torch import nn , optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from Tooth_2Ddataloader import Tooth , CBCT_path ,DH_img_path ,DH_label_path,all_data_path,all_label_path
from Tooth_2Dnet import Resnet18 , UNet , config
from uct_nets.UCTransNet import UCTransNet
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import time
import torch.distributed as dist
import os
from PIL import Image


global step 
step = 0

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)

batchsz2=1
lr=1e-3
epochs=5

path_SRT="/mnt"
path_save_mask = "/mnt/LLSRT/test_result"

para_load=path_SRT+"/LLSRT/para/Unet_focal1.2.mdl"
config3=config()
model4 = UCTransNet(config3,n_channels=1,n_classes=1,img_size=256)
model4.to(device)
model4 = torch.nn.parallel.DistributedDataParallel(model4, device_ids=[local_rank], output_device=local_rank)

torch.cuda.empty_cache()

tooth_DH_db_eval=Tooth(all_data_path,all_label_path,(256,256,256),mode='eval',load_mode="all")
tooth_DH_loader_eval = DataLoader(tooth_DH_db_eval,batch_size=batchsz2,shuffle=False,num_workers=1,pin_memory=True,
                                  sampler=DistributedSampler(tooth_DH_db_eval,shuffle=False))
def Unet2_test_main():
    # model3.load_state_dict(torch.load(para_load_u2))
    model4.load_state_dict(torch.load(para_load))
    model4.eval()
    with torch.no_grad():
        global step 
        step = 0
        for x,y,label in tooth_DH_loader_eval:
            x,y,label=x.to(device),y.to(device),label.to(device)
            mask = model4(x)
            mask = torch.where(mask>0.5,255.,0.).detach()
            array = mask.view(256,256).cpu().numpy()
            im = Image.fromarray(array).convert("L")
            im.save("/mnt/LLSRT/test_result/test_"+str(step)+"_.jpeg")
            step+=1
    print("__test finish and saved__")
        
if __name__=='__main__':
    Unet2_test_main()
    
"""
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 /mnt/LLSRT/Tooth_2D_test.py
"""