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
from uct_nets.UNetp import UNetp
from uct_nets.UNetpp import UNetpp
from uct_nets.DCUnet import DC_Unet
from UNet2D_BraTs.unet import UnetB
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import time
import torch.distributed as dist

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

batchsz1=4
batchsz2=8
lr=1e-3
epochs=5
classes=6
path_SRT="/mnt"

para_save=path_SRT+"/LLSRT/para/q_resnet1.1.mdl"
para_load=path_SRT+"/LLSRT/para/q_resnet1.1.mdl"
para_save_u3=path_SRT+"/LLSRT/para/UNet/DCUNet/q_Unet_GPU.pth"
para_load_u3=path_SRT+"/LLSRT/para/UNet/DCUNet/q_Unet_GPU.pth" #now 6 attention

#cyb
path_config = path_SRT + "/LLSRT/img"

# 1.1 class=6 ratio = (1,1)
# 1.2 class=7 ratio = (1,1) （abandon）
# 1.3 class=6 ratio = (1,1.5)
config3=config(1,1.5)
# model4 = UCTransNet(config3,n_channels=1,n_classes=classes,img_size=320)
# model4 = UNetpp(n_channels=1,n_classes=classes)
model4 = DC_Unet(in_channels=1, out_channels=classes,use_dropout=True)
# UNet:UNetc  UNet+:UNetp  UNet++:UNetpp
model4.to(device)
if multiGPU :
    model4 = torch.nn.parallel.DistributedDataParallel(model4, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

optimizer4 = optim.Adam(model4.parameters(),lr=lr,weight_decay=1e-6)

criteon1 = nn.CrossEntropyLoss().to(device)
criteon2 = nn.MSELoss().to(device)
criteon31 = DiceLoss0().to(device)
criteon32 = nn.BCEWithLogitsLoss().to(device)
criteon33 = Multi_FocalLoss(device).to(device)
criteon3 = FocalLoss().to(device)

#torch.manual_seed(8848)
print(sum(p.numel() for p in model4.parameters()))
torch.cuda.empty_cache()

tooth_DH_db=Tooth(all_queue_data_path,all_queue_label_path,(320,320,320),mode='train',load_mode="all",num_class=classes)
if multiGPU :
    tooth_DH_loader = DataLoader(tooth_DH_db,batch_size=batchsz1,num_workers=3,pin_memory=True,
                                sampler=DistributedSampler(tooth_DH_db,shuffle=True))
else:
    tooth_DH_loader = DataLoader(tooth_DH_db,batch_size=batchsz1,shuffle=False,num_workers=3,pin_memory=True,
                                )#sampler=DistributedSampler(tooth_DH_db,shuffle=False))
tooth_DH_db_eval=Tooth(all_queue_data_path,all_queue_label_path,(320,320,320),mode='eval',load_mode="all",num_class=classes)
tooth_DH_loader_eval = DataLoader(tooth_DH_db_eval,batch_size=batchsz2,shuffle=False,num_workers=2,pin_memory=True,
                                  )#sampler=DistributedSampler(tooth_DH_db_eval,shuffle=False))

def Unet2_main():
    # model3.load_state_dict(torch.load(para_load_u2))
    
    # model4.load_state_dict(torch.load(para_load_u3))
    torch.cuda.empty_cache()
    
    if torch.cuda.device_count() > 1:
        print("we can use ", torch.cuda.device_count(), "GPUs.")
    
    epochs=32*8 + 1 
    best_acc = 0.00
    for epoch in range(epochs):
        if multiGPU :
            tooth_DH_loader.sampler.set_epoch(epoch)
        
        if epoch%4==0:
            print("__epoch start test__")
            model4.eval()
            with torch.no_grad():
                step=0
                total_acc = 0
                total_and = 0
                total_or = 0
                total_dix = 0
                total_y = 0
                total_sensitive = 0
                color = tooth_DH_db_eval.color
                for idx,(x,y,label) in enumerate(tooth_DH_loader_eval):
                    
                    if idx % 1 == 0:
                        x , y , label = x.to(device) ,  y.to(device) , label.to(device)
                        x0=F.softmax(model4(x),dim=1)
                        mask0_value = torch.max(x0,dim=1)[0].data.unsqueeze(1)
                        mask0 = torch.max(x0,dim=1)[1].unsqueeze(1)

                        #print(mask0_value.sum())
                        #mask0 = torch.ones_like(y)*(epoch+1)
                        color_id = 0
                        color_lim = 0.4
                        x0 = ((x0 == mask0_value)&(mask0_value>color_lim)).to(dtype=torch.int64)
                        for cc in color:
                            mask_c1 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[0],0)
                            mask_c2 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[1],0)
                            mask_c3 = torch.where((mask0==color_id)&(mask0_value>color_lim),cc[2],0)
                            if color_id == 0 :
                                mask = torch.cat((mask_c1,mask_c2,mask_c3),dim=1)       
                            else:
                                mask += torch.cat((mask_c1,mask_c2,mask_c3),dim=1)                
                            color_id += 1
                            
                        total_dix += x.size(0)
                        z = ( y + 1 )/2
                        y = y.to(torch.int64)
                        total_and += torch.eq(x0,z).sum()
                        total_or += torch.ne(x0,y).sum() + torch.eq(x0,z).sum()
                        total_y += torch.where(y>0.5,1.,0.).sum()
                        
                        #print("id=1 sum=",mask[1].sum())
                        #print("id=3 sum=",mask[3].sum())
                        #print("id=5 sum=",mask[5].sum())
                        if step % 10 == 0:
                            plt.figure(figsize=(40,40))
                            for id in range(mask.size(0)):
                                plt.subplot(4,4,2*id+1)
                                plt.imshow((mask[id].cpu()).transpose(0,-1).transpose(0,1).numpy().reshape(320,320,-1).astype(np.uint8))
                                plt.title("mask id = "+str(id))
                                plt.subplot(4,4,2*id+2)
                                plt.title("label id = "+str(id))
                                plt.imshow((label[id].cpu()).transpose(0,-1).transpose(0,1).numpy().reshape(320,320,-1).astype(np.uint8))
                            plt.tight_layout()
                            plt.savefig(path_SRT+"/LLSRT/img/pic_"+str(step//10)+"_.jpg")
                            plt.close() 
                        step+=1
                total_acc = total_and/total_or
                total_sensitive = total_and /total_y
                dice_acc = total_acc*2/(1+total_acc)
                if dice_acc > best_acc:
                    best_acc = dice_acc
                    torch.save(model4.state_dict(),para_save_u3) 
                print("epoch:<< ",epoch," >>  dice = ",float(dice_acc)," sensitive = ",float(total_sensitive),
                      " miou = ",float(total_acc),"  best_acc = ",best_acc)
                
                #cyb
                with open(path_config +'/result.txt', 'a', encoding= 'utf-8') as f:
                    f.write("epoch:<< " + str(epoch) + " >>  miou = " + str(float(total_acc)) + 
                            " sensitive = " + str(float(total_sensitive)) + "  best_acc = " + str(float(best_acc))+
                            "  dice = " + str(float(dice_acc))+"\n")
                    f.close()          
            # if epoch > 0:
            #     torch.save(model4.state_dict(),para_save_u3) 
            print("__test finish and saved__")
        
        print("__epoch start train__")
        model4.train()
            
        for batchidx,(x,y) in enumerate(tooth_DH_loader):
            x,y=x.to(device),y.to(device)
            mask=model4(x)
            loss4=criteon33(mask,y)*10000
            optimizer4.zero_grad()
            loss4.backward()
            optimizer4.step()
            if batchidx  % 40 == 0:
                print("   batch idx: ",batchidx,loss4.item())               
        print("epoch: ",epoch,loss4.item())    
        

            

if __name__=='__main__':
    Unet2_main()
    
"""
命令行之星下面语句:(nproc_per_node=GPU个数)
python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 /mnt/LLSRT/Tooth_queue_train3.py
"""