import torch
from torch import nn , optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from Tooth_2Ddataloader import Tooth , CBCT_path ,DH_img_path ,DH_label_path,all_data_path,all_label_path
from Tooth_2Dnet import Resnet18 , UNet , config
from Tooth_2DLOSS import DiceLoss0,WeightedFocalLoss , FocalLoss
from uct_nets.UCTransNet import UCTransNet
from uct_nets.UNet import UNetc
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

batchsz1=12
batchsz2=8
lr=1e-3
epochs=5

path_SRT="/mnt"

para_save=path_SRT+"/LLSRT/para/resnet1.1.mdl"
para_load=path_SRT+"/LLSRT/para/resnet1.1.mdl"
para_save_u=path_SRT+"/LLSRT/para/Unet1.1.mdl"
para_load_u=path_SRT+"/LLSRT/para/Unet1.1.mdl"
para_save_u2=path_SRT+"/LLSRT/para/Unet_crop1.1.mdl"
para_load_u2=path_SRT+"/LLSRT/para/Unet_crop1.1.mdl"
para_save_u3=path_SRT+"/LLSRT/para/Unet_focal1.3.mdl"
para_load_u3=path_SRT+"/LLSRT/para/Unet_focal1.2.mdl"

config3=config()
model4 = UCTransNet(config3,n_channels=1,n_classes=1,img_size=256)

model = Resnet18(2).to(device)
model2 = UNet(1,1).to(device)
model3 = UCTransNet(config3,n_channels=1,n_classes=1,img_size=256).to(device)
model4.to(device)
if multiGPU :
    model4 = torch.nn.parallel.DistributedDataParallel(model4, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
optimizer2 = optim.Adam(model2.parameters(),lr=lr,weight_decay=1e-6)
optimizer3 = optim.Adam(model3.parameters(),lr=lr,weight_decay=1e-6)
optimizer4 = optim.Adam(model4.parameters(),lr=lr,weight_decay=1e-6)

criteon = nn.CrossEntropyLoss().to(device)
criteon2 = nn.MSELoss().to(device)
criteon31 = DiceLoss0().to(device)
criteon32 = nn.BCEWithLogitsLoss().to(device)
criteon33 = FocalLoss().to(device)

torch.manual_seed(114514)

torch.cuda.empty_cache()

tooth_DH_db=Tooth(all_data_path,all_label_path,(256,256,256),mode='train',load_mode="all")
if multiGPU :
    tooth_DH_loader = DataLoader(tooth_DH_db,batch_size=batchsz1,num_workers=3,pin_memory=True,
                                sampler=DistributedSampler(tooth_DH_db,shuffle=True))
else:
    tooth_DH_loader = DataLoader(tooth_DH_db,batch_size=batchsz1,shuffle=False,num_workers=3,pin_memory=True,
                                )#sampler=DistributedSampler(tooth_DH_db,shuffle=False))
tooth_DH_db_eval=Tooth(all_data_path,all_label_path,(256,256,256),mode='eval',load_mode="all")
tooth_DH_loader_eval = DataLoader(tooth_DH_db_eval,batch_size=batchsz2,shuffle=False,num_workers=2,pin_memory=True,
                                  )#sampler=DistributedSampler(tooth_DH_db_eval,shuffle=False))

def Resnet_main():
    model.load_state_dict(torch.load(para_load))
    best_acc=0
    for epoch in range(epochs):
        model.train()
        for batchidx,(x,y,label) in enumerate(tooth_DH_loader):
            x,y,label=x.to(device),y.to(device),label.to(device)
            logits=model(x)
            loss=criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("   batch idx: ",batchidx,loss.item())               
        print("epoch: ",epoch,loss.item())    
        
        if epoch%4==1:
            model.eval()
            with torch.no_grad():
                total_correct=0.
                total_num=0.
                pred_sum=0
                label_sum=0
                for x,y,label in tooth_DH_loader_eval:
                    x,y,label=x.to(device),y.to(device),label.to(device)
                    
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    if epoch%4==1:
                        plt.figure(figsize=(16,12))
                        px=np.arange(0,label.size(0),1)
                        plt.plot(px,pred,c='r')
                        plt.plot(px,label,c='b')
                        plt.show()
                    labelx=label/2+0.5
                    total_correct+=torch.eq(pred,labelx).float().sum()
                    pred_sum+=pred.sum()
                    label_sum+=label.sum()
                
                total_num=max(pred_sum,label_sum)  
                acc=total_correct/total_num
                print("test:",epoch," acc:",float(acc)*100,"%")
                if acc >= best_acc:
                    best_acc=acc
                    print("best_acc = ",float(best_acc)*100,"%")
                    torch.save(model.state_dict(),para_save)  
def Unet_main():
    #model2.load_state_dict(torch.load(para_load_u))
    epochs=1
    for epoch in range(epochs):
        model2.train()
        for batchidx,(x,y,label) in enumerate(tooth_DH_loader):
            x,y,label=x.to(device),y.to(device),label.to(device)
            mask=model2(x)
            loss2=criteon2(y,mask)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            print("   batch idx: ",batchidx,loss2.item())               
        print("epoch: ",epoch,loss2.item())    
        
        if epoch%4==0:
            model2.eval()
            with torch.no_grad():
                step=0
                for x,y,label in tooth_DH_loader_eval:
                    x,y,label=x.to(device),y.to(device),label.to(device)
                    
                    mask = model2(x)
                    plt.figure(figsize=(30,30))
                    for id in range(mask.size(0)):
                        plt.subplot(4,4,2*id+1)
                        plt.imshow(mask[id].view(256,256).cpu().numpy(),cmap="Greys_r")
                        plt.title("mask id = "+str(id))
                        plt.subplot(4,6,2*id+2)
                        plt.title("label id = "+str(id))
                        plt.imshow(y[id].view(256,256).cpu().numpy(),cmap="Greys_r")
                    plt.savefig(path_SRT+"/LLSRT/img/pic_"+str(step)+"_.jpg")
                    step+=1
                torch.save(model2.state_dict(),para_save_u)      
def Unet2_main():
    # model3.load_state_dict(torch.load(para_load_u2))
    
    # model4.load_state_dict(torch.load(para_load_u3))
    torch.cuda.empty_cache()
    
    if torch.cuda.device_count() > 1:
        print("we can use ", torch.cuda.device_count(), "GPUs.")
        
    epochs=32*8 + 1 
    best_acc = 0.70
    for epoch in range(epochs):
        
        print("__epoch start train__")
        model4.train()
        if multiGPU :
            tooth_DH_loader.sampler.set_epoch(epoch)
            
        for batchidx,(x,y,label) in enumerate(tooth_DH_loader):
            x,y,label=x.to(device),y.to(device),label.to(device)
            mask=model4(x)
            loss4=criteon33(mask,y)*10000
            optimizer4.zero_grad()
            loss4.backward()
            optimizer4.step()
            if batchidx%5 == 0:
                print("   batch idx: ",batchidx,loss4.item())               
        print("epoch: ",epoch,loss4.item())    
        
        if epoch%4==0:
            print("__epoch start test__")
            model4.eval()
            with torch.no_grad():
                step=0
                total_acc = 0
                total_and = 0
                total_or = 0
                total_dix = 0
                for x,y,label in tooth_DH_loader_eval:
                    x,y,label=x.to(device),y.to(device),label.to(device)
                    mask = model4(x)
                    mask = torch.where(mask>0.5,1.,0.)
                    
                    total_dix += x.size(0)
                    z = ( y + 1 )/2
                    total_and += torch.eq(mask,z).sum()
                    total_or += torch.ne(mask,y).sum() + torch.eq(mask,z).sum()
                    
                    plt.figure(figsize=(40,20))
                    for id in range(mask.size(0)):
                        plt.subplot(4,4,2*id+1)
                        plt.imshow(mask[id].view(256,256).cpu().numpy(),cmap="Greys_r")
                        plt.title("mask id = "+str(id))
                        plt.subplot(4,4,2*id+2)
                        plt.title("label id = "+str(id))
                        plt.imshow(y[id].view(256,256).cpu().numpy(),cmap="Greys_r")
                    plt.tight_layout()
                    plt.savefig(path_SRT+"/LLSRT/img/pic_"+str(step)+"_.jpg")
                    step+=1
                total_acc = total_and/total_or
                if total_acc > best_acc:
                    best_acc = total_acc
                    torch.save(model4.state_dict(),para_save_u3) 
                print("epoch:<< ",epoch," >>  acc = ",float(total_acc),"  best_acc = ",best_acc)
            print("__test finish and saved__")
            

if __name__=='__main__':
    Unet2_main()
    
"""
命令行之星下面语句：（nproc_per_node=GPU个数）
python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 /mnt/LLSRT/Tooth_2Dtrain.py
"""