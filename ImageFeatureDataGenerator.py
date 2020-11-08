
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH="C:/Users/D-blue/Desktop/Humen_Behaviour/project/"
TEXT_LENGTH=75
TEXT_HIDDEN=256
"""
read text file, find corresponding image path
"""
def load_data():
    data_set=dict()
    for dataset in ["train"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[2]
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set[int(image)]={"text":sentence,"group":group}
    for dataset in ["test","valid"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set[int(image)]={"text":sentence,"group":group}
    return data_set

data_set=load_data()
"""
load image data
"""
image_feature_folder="image_feature_data_temp"
# pretrain dataloader
class pretrain_data_set(Dataset):
    def __init__(self, data):
        self.data=data
        self.image_ids=list(data.keys())
        for id in data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_data/",str(id)+".jpg")
    # load image
    def __image_loader(self,id):
        path=self.data[id]["image_path"]
        img_pil =  PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        img_tensor = transform(img_pil)
        return img_tensor
    def __getitem__(self, index):
        id=self.image_ids[index]
        image=self.__image_loader(id)
        return id,image
    def __len__(self):
        return len(self.image_ids)

sub_image_size=32 #448/14
sub_graph_preprocess = transforms.Compose([
    transforms.ToPILImage(mode=None),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
all_pretrain_dataset=pretrain_data_set(data_set)
"""
generate data
"""
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
def resnet50_predictor():
    # extract the input for last fc layer in resenet50
    resnet50=torchvision.models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    resnet50.fc = Identity()
    resnet50 = resnet50.to(device)
    resnet50.eval()
    # save the output in .npy file
    resnet50_output_path=os.path.join(WORKING_PATH,image_feature_folder)
    if not os.path.exists(resnet50_output_path):
        os.makedirs(resnet50_output_path)
    with torch.no_grad():
        total=len(all_pretrain_loader)*all_pretrain_loader.batch_size
        count=0
        time_s=time.perf_counter()
        for img_index,img in all_pretrain_loader:

            # seperate img(448,448) into 14*14 images with size (32,32)
            # [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            # [14,15,16,17,18,................]
            # [28,...]
            # ...
            # [182,....,195]
            sub_img_output=list()
            for column in range(14):
                for row in range(14):
                    # resize image from (32,32) to (256,256)
                    sub_image_original=img[:,:,sub_image_size*row:sub_image_size*(row+1),sub_image_size*column:sub_image_size*(column+1)]
                    sub_image_normalized=torch.stack(list(map(lambda image:sub_graph_preprocess(image),sub_image_original)),dim=0)
                    output=resnet50(sub_image_normalized.to(device))
                    sub_img_output.append(output.to("cpu").numpy())
            sub_img_output=np.array(sub_img_output).transpose([1,0,2])
            # save averaged attribute to "resnet50_output", same name as the image
            for index,sub_img_index in enumerate(img_index):
                np.save(os.path.join(resnet50_output_path,str(sub_img_index.item())),sub_img_output[index])
            time_e=time.perf_counter()
            count+=all_pretrain_loader.batch_size
            total_time=time_e-time_s
            print(f"Completed {count}/{total} time left={int((total-count)*total_time/count/60/60)}:{int((total-count)*total_time/count/60%60)}:{int((total-count)*total_time/count%60)} speed={round(total_time/count,3)}sec/image")


# 32 is the minimum batch size can achieve best performance
all_pretrain_loader = DataLoader(all_pretrain_dataset,batch_size=64)

# it will take really long time to run...
resnet50_predictor()
"""
test the image split
"""
# if __name__ == "__main__":
#     # can be used to create image feature data
#     # resnet50_predictor()
#     for img_index,img in all_loader:
#         temp_img=img
#         print(img[0].size())
#         plt.imshow(img[0].permute(1,2,0))
#         plt.show()
#         print("======================================")
#         # try to seperate
#         for column in range(14):
#             for row in range(14):
#                 sub_index=row*14+column
#                 sub_image_original=img[0][:,sub_image_size*row:sub_image_size*(row+1),sub_image_size*column:sub_image_size*(column+1)]
#                 sub_image_normalized=sub_graph_preprocess(sub_image_original)
#                 # show original sub image
#                 plt.imshow(sub_image_original.permute(1,2,0))
#                 plt.show()
#                 # show normalized sub image
#                 plt.imshow(sub_image_normalized.permute(1,2,0))
#                 plt.show()
#                 print(sub_index)
#                 print(sub_image_original.size())
#                 print(sub_image_normalized.size())
#                 break
#             break
#         break


