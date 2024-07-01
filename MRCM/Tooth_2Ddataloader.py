import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os,csv,glob
import pydicom
import SimpleITK as sitk
from PIL import Image

CBCT_path="/mnt"
DH_img_path="data/20160605083002_data"
DH_label_path="label/20160605083002_label"
all_data_path="/data"
all_label_path="/label"
all_queue_data_path="/queue_data"
all_queue_label_path="/queue_label"

class Tooth(Dataset):
    def __init__(self,img_relative_root,label_relative_root,size=(256,256),mode="train",origin_root=CBCT_path
                 ,load_mode="all",index_data=False,num_class=1):
        super(Dataset,self).__init__()
        if load_mode == "image_label" :
            ####### essential parameters #######
            self.img_path=origin_root+img_relative_root
            self.label_path=origin_root+label_relative_root             
            self.size=size
            self.index_data=index_data
            self.mode=mode
            ####### load filename #######
            self.images=[]
            self.labels=[]
            for id , img_name in enumerate(sorted(os.listdir((os.path.join(self.img_path))))):
                if img_name[-3:]=='dcm' :
                    self.images.append(self.img_path+"/"+img_name)
            for id , label_name in enumerate(sorted(os.listdir((os.path.join(self.label_path))))):
                if (label_name[-3:]=='dcm') or (label_name[-3:]=='bmp') :
                    self.labels.append(self.label_path+"/"+label_name)
            # print(len(self.images),len(self.labels))
            # print(self.images,self.labels)
            self.total_len=self.__len__()
            if mode == "train":
                self.images=self.images[80:]
                self.labels=self.labels[80:]
            elif mode == "eval":
                self.images=self.images[:80]
                self.labels=self.labels[:80]
        if load_mode == "all" :
            ####### essential parameters #######
            self.all_img_path=origin_root+img_relative_root
            self.all_label_path=origin_root+label_relative_root
            self.size=size
            self.index_data=index_data
            self.mode=mode
            self.num_class = num_class
            self.color=[
                 [254,0,0],
                 [254,254,0],
                 [127,254,0],
                 [0,254,64],
                 [0,254,254],
                 [254,0,254]
                 ]
   
            ####### load filename #######
            self.images=[]
            self.labels=[]
            for folder_name in sorted(os.listdir((os.path.join(self.all_img_path)))):
                        
                load_lo = 0
                load_hi = 320
                if folder_name[0] == 'M' :
                    load_lo = 0
                    load_hi = 640
                for id , img_name in enumerate(sorted(os.listdir((os.path.join(self.all_img_path+"/"+folder_name))))):
                    if (img_name[-3:]=='dcm') & ((load_lo<=id)&(id<=load_hi)):
                        self.images.append(self.all_img_path+"/"+folder_name+"/"+img_name)
            for folder_name in sorted(os.listdir((os.path.join(self.all_label_path)))):  
                            
                load_lo = 0
                load_hi = 320
                if folder_name[0] == 'M' :
                    load_lo = 0
                    load_hi = 640
                for id , label_name in enumerate(sorted(os.listdir((os.path.join(self.all_label_path+"/"+folder_name))))):
                    if (label_name[-3:]=='dcm' or label_name[-3:]=='bmp') & ((load_lo<=id)&(id<=load_hi)):
                        self.labels.append(self.all_label_path+"/"+folder_name+"/"+label_name)
            #print(len(self.images),len(self.labels))
                #print(self.images,self.labels)
            self.total_len=self.__len__()
            eval_line = 320 * 2
            if mode == "train":
                self.images=self.images[eval_line:]
                self.labels=self.labels[eval_line:]
            elif mode == "eval":
                self.images=self.images[:eval_line]
                self.labels=self.labels[:eval_line]    
                
    def crop_img(self,img,Point_lo,Point_hi):
        img = img [...,Point_lo[0]:Point_hi[0],Point_lo[1]:Point_hi[1]]
        return img
  
    def __len__(self):
        return len(self.images)
    
    def get_img(self, idx , resize_img = 512):
        if (idx < 0) :
            idx=0
        if (idx >= self.__len__()):
            idx=self.__len__()-1
        img = self.images[idx]
        img_tf=transforms.Compose([
        transforms.Resize((resize_img,resize_img)),
        transforms.ToTensor(),
        ])
        img_0 = pydicom.read_file(img).pixel_array
        img_img = np.float32( img_0 + 1024)/4096
        img_img = Image.fromarray(img_img)
        img_data = img_tf(img_img) 
        point_crop = (100,200)
        img_data = self.crop_img(img_data,(point_crop[0],point_crop[1]),(point_crop[0]+256,point_crop[1]+256))
        return img_data
    
    def __getitem__(self, idx ):
        #index-[0~len]
        #self.images self.labels
        seed = torch.random.seed()

        img , label = self.images[idx] , self.labels[idx]
        
        """        print("========= test ==========")
        print("== loading " + img + " ==")
        test_img=pydicom.read_file(img)
        print(test_img.pixel_array.shape)
        print("========= done ==========")  
        
        print("========= test ==========")
        print("== loading " + label + " ==")
        test_img=pydicom.read_file(label)
        print(test_img.pixel_array.shape)
        print("========= done ==========")   """        
        
        resize_img = 620
        if self.mode == "eval":
            img_tf=transforms.Compose([
                transforms.Resize((resize_img,resize_img)),
                transforms.ToTensor(),
            ])
        else:
            img_tf=transforms.Compose([
                transforms.Resize((resize_img,resize_img)),
                transforms.CenterCrop(560),
                transforms.RandomCrop(320),
                transforms.ToTensor(),
            ])
        #print(img)
        torch.random.manual_seed(seed)
        # ds = pydicom.read_file(img,force=True)
        # ds.file_meta.TransferSyntaxUID =pydicom.uid.ImplicitVRLittleEndian
        # img_0 = ds.pixel_array        
        ds = sitk.ReadImage(img)
        img_0 = sitk.GetArrayFromImage(ds)
        img_img = np.float32( img_0 + 1024)/4096
        img_shape = img_img.shape
        img_img = Image.fromarray(img_img.reshape(img_shape[-2],img_shape[-1]))
        img_data = img_tf(img_img) 
        
        if self.num_class==1:
            label_tf=transforms.Compose([
                #transforms.Resize((resize_img,resize_img)),
                transforms.ToTensor(),
            ])
            label_0 = pydicom.read_file(label).pixel_array

            # if img_0.shape != label_0.shape :
            #     print ("Not match : id = ",idx," img_size=",img_0.shape ," label_size=", label_0.shape)
            # if (img_0.shape[-1] != 640) or (label_0.shape[-1] != 640):
            #     print ("Not match 620 : id = ",idx)        
            label_img = np.float32( label_0 - img_0 )/4096
            label_img = Image.fromarray(label_img)
            label_data = label_tf(label_img) 
            label_data = torch.where(label_data>0.31,1.,0.)
            #print(label_data)
            lim = 4
            if bool(torch.sum(label_data)-lim > 0):
                judge = torch.tensor(1)
            else:
                judge = torch.tensor(0)  
                
            point_crop = (100,200)
            img_data = self.crop_img(img_data,(point_crop[0],point_crop[1]),(point_crop[0]+256,point_crop[1]+256))
            label_data = self.crop_img(label_data,(point_crop[0],point_crop[1]),(point_crop[0]+256,point_crop[1]+256)) 
            
        elif ( self.num_class >= 2 ):
            if self.mode == "eval":
                label_rgb_tf=transforms.Compose([
                    lambda x:Image.open(x).convert("RGB"),
                    transforms.Resize((resize_img,resize_img)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                ])       
            else:
                label_rgb_tf=transforms.Compose([
                    lambda x:Image.open(x).convert("RGB"),
                    transforms.Resize((resize_img,resize_img)),
                    transforms.CenterCrop(560),
                    transforms.RandomCrop(320),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                ])        
        
            width_lim = 0.9
            color_id = 0
            torch.random.manual_seed(seed)
            label = label_rgb_tf(label) * 255
            label_color_id = torch.zeros_like(img_data,dtype=torch.int64)
            for cc in self.color:
                cc = torch.tensor(cc,dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                label_c = label - cc
                label_c1 = torch.where((width_lim>label_c[0])&(label_c[0]>-width_lim),1.,0.)
                label_c2 = torch.where((width_lim>label_c[1])&(label_c[1]>-width_lim),1.,0.)
                label_c3 = torch.where((width_lim>label_c[2])&(label_c[2]>-width_lim),1.,0.)
                label_out = torch.where((label_c1==1)&(label_c2==1)&(label_c3==1),1.,0.).unsqueeze(0)
                label_color_id += torch.where((label_c1==1)&(label_c2==1)&(label_c3==1),color_id,0).unsqueeze(0)
                if color_id == 0 :
                    label_data = label_out
                else:
                    label_data = torch.cat((label_data,label_out),dim=0)    
                color_id+=1
                
            # background = 1./self.num_class
            # label_data = label_data*(1-background)+background
            #zero
            label_zero_c1 = torch.where((width_lim>label[0])&(label[0]>-width_lim),1.,0.)
            label_zero_c2 = torch.where((width_lim>label[1])&(label[1]>-width_lim),1.,0.)
            label_zero_c3 = torch.where((width_lim>label[2])&(label[2]>-width_lim),1.,0.)
            label_zero_out = torch.where((label_zero_c1==1)&(label_zero_c2==1)&(label_zero_c3==1),1./self.num_class,0.).unsqueeze(0)
            label_data += label_zero_out      
            #white
            label_w = label - torch.tensor([255,255,255],dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            label_white_c1 = torch.where((width_lim>label_w[0])&(label_w[0]>-width_lim),1.,0.)
            label_white_c2 = torch.where((width_lim>label_w[1])&(label_w[1]>-width_lim),1.,0.)
            label_white_c3 = torch.where((width_lim>label_w[2])&(label_w[2]>-width_lim),1.,0.)
            label_white_out = torch.where((label_white_c1==1)&(label_white_c2==1)&(label_white_c3==1),1./self.num_class,0.).unsqueeze(0)            
            label_data += label_white_out
            lim = 4
            #print(label_data)
            #label_data = label_color_id
            if bool(torch.sum(label_data)-lim > 0):
                judge = torch.tensor(1)
            else:
                judge = torch.tensor(0) 
                
            if self.mode == "eval":
                point_crop = (85,165)
                cube_size = 320
                img_data = self.crop_img(img_data,(point_crop[0],point_crop[1]),(point_crop[0]+cube_size,point_crop[1]+cube_size))
                label_data = self.crop_img(label_data,(point_crop[0],point_crop[1]),(point_crop[0]+cube_size,point_crop[1]+cube_size))
                label = self.crop_img(label,(point_crop[0],point_crop[1]),(point_crop[0]+cube_size,point_crop[1]+cube_size))           

            
        """img_data = torch.cat((self.get_img(idx-5),
                        img_data,
                            self.get_img(idx+5),
                            ),dim=0)"""
        if self.mode == "eval":
            return img_data , label_data , label 
        elif self.mode == "loader":
            return img_data , label_data , label , judge            
        else:
            return img_data , label_data 
        
if __name__=='__main__':
    #test_img=pydicom.read_file("D:/CBCT-AI/CBCT-AI_标注神经管/demo/denghao/20160605083002-0-1-10/Slice_0000.dcm")
    #print(test_img.pixel_array.shape)
    classes = 6
    tooth_DH_db=Tooth(all_queue_data_path,all_queue_label_path,(256,256,256),load_mode="all",num_class=classes,mode="loader")
    tooth_DH_loader = DataLoader(tooth_DH_db,batch_size=1,shuffle=False)
    print("======= loading done =======")
    """
    for x , y in tooth_DH_loader:
        print(x.shape,y.shape)"""
    j_sum = 0
    n_sum = 0
    for idx,(x,y,y0,j) in enumerate(tooth_DH_loader):
        #print(idx)
        #print(x.shape,y.shape)
        if j == 1 :
            if (j_sum%50==1) :
                #y1=torch.transpose(y,1, 2)
                #y1=torch.transpose(y1, 2, 3)
                out_size=320
                print('idx: ',idx,'  sample:',x.shape,y.shape)
                plt.figure(figsize=(20,20))
                plt.subplot(3,3,1)
                plt.imshow(x.numpy().reshape(out_size,out_size), cmap='gray')
                #plt.imshow(x.numpy().reshape(620,620),cmap='Greys_r')
                for i in range(classes):
                    plt.subplot(3,3,i+2)
                    plt.imshow(y[0][i].numpy().reshape(out_size,out_size), cmap='gray')
                    # print(y[0][i].numpy().sum()/(out_size*out_size))
                plt.subplot(3,3,9)
                plt.imshow((y0[0]).transpose(0,-1).transpose(0,1).numpy().reshape(out_size,out_size,-1).astype(np.uint8))
                # plt.subplot(3,3,9)
                # plt.imshow(x.numpy().reshape(out_size,out_size),cmap='Greys_r')
                # plt.subplot(3,3,9)
                # plt.imshow((y0[0]+x[0]*200).transpose(0,-1).transpose(0,1).numpy().reshape(out_size,out_size,-1).astype(np.uint8))
                plt.tight_layout()
                plt.savefig("/mnt/LLSRT/img/img_dataset/label_"+str(idx)+".jpeg")
                plt.clf()
                plt.close()
                #plt.imshow(y.numpy().reshape(620,620),cmap='Greys_r')
            j_sum += 1
        n_sum += 1 
    print(j_sum,n_sum)

    