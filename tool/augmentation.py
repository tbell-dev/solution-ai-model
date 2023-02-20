import albumentations as A
import random
import cv2
import numpy as np
import os

from glob import glob
import json
import argparse
from pycocotools.coco import COCO

from PIL import Image

from datetime import date
from modules.formatter import coco_style_gen

class augmentator:
    # user 정보를 활용하여 dataset의 위치를 특정 -> image와 json 파일을 불러옴
    def __init__(self,user_dataset_dir,output_dir,labeling_type = "seg",iter = 20):
        self.dataset_path = user_dataset_dir
        self.output_dir = output_dir
        self.task = labeling_type
        self.iter = iter
        # self.img_list = [i for i in glob(self.dataset_path+"/*.jpg")]
        self.img_list = [i for ext in ('*.jpg', '*.jpeg', '*.JPEG',"*.JPG") for i in glob(self.dataset_path+"/"+ext)]
        imsize_list = [cv2.imread(i).shape for i in self.img_list]
        width_list = [imsize_list[i][1] for i in range(len(imsize_list))]
        height_list = [imsize_list[i][0] for i in range(len(imsize_list))]
        self.min_image_size=(min(width_list),min(height_list))
        self.json_data = [i for i in glob(self.dataset_path+"/*.json")][0]
        self.aug = self.load_pipeline()
        self.cats = []
        self.image_id = 0
        self.form = coco_style_gen()
        self.all_annos = 0
        self.start()
    
    def load_pipeline(self):
        if self.task == "seg":
            transform = A.Compose([
            # A.RandomCrop(width=int(self.min_image_size[0]*0.9), height=int(self.min_image_size[1]*0.9),p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), contrast_limit=(-0.2, 0.3), p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.5),
                A.Blur(p=0.5),
                A.CLAHE(p=0.5)
            ], p=0.8)
            ])
        elif self.task == "od":
            transform = A.Compose([
            # A.RandomCrop(width=int(self.min_image_size[0]*0.9), height=int(self.min_image_size[1]*0.9),p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), contrast_limit=(-0.2, 0.3), p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.5),
                A.Blur(p=0.5),
                A.CLAHE(p=0.5)
            ], p=0.8)
            ],bbox_params=A.BboxParams(format='coco',label_fields = ['class_labels']))
        random.seed(42)
        
        return transform
    # 사용자가 업로드한 라벨 데이터는 coco 형식의 json 파일로써 한개만 존재한다고 가정       
    def start(self):
        coco_annotation = COCO(self.json_data)
        with open(self.json_data, 'r') as f: coco_data = json.load(f)
        imgids = coco_annotation.getImgIds()
        self.cats = coco_annotation.loadCats(coco_annotation.getCatIds())
        for i in range(len(imgids)):
            mask_list=[]
            cat_list = []
            
            box_list = []
            cls_name = []
            
            ann_ids = coco_annotation.getAnnIds(imgIds=[imgids[i]], iscrowd=None)
            anns = coco_annotation.loadAnns(ann_ids)
            
            img_file_name = [imginfo["file_name"].split("/")[-1] for imginfo in coco_data["images"] if imginfo["id"] == imgids[i]][0]
            for j in range(len(anns)):
                if self.task == 'seg':
                    cat_list.append(anns[j]["category_id"])
                    mask_list.append(coco_annotation.annToMask(anns[j]))
                elif self.task == 'od':
                    box_list.append(anns[j]["bbox"])
                    cat_list.append(anns[j]["category_id"])
                    cls_name.append([cat["name"] for cat in self.cats if anns[j]["category_id"] == cat["id"]][0])
                
                
            # images = [img1,img2,.....] / masks = [[mask1,mask2,mask3,....],[mask1,mask2,mask3,....],[mask1,mask2,mask3,....],......] masks의 각 인덱스는 images의 각 인덱스와 매핑됨,
            # masks[n] = [mask1,mask2,mask3,...] : img1 에 포함되어있는 class들의 마스크 즉, mask1 ~ maskn 은 cat_list의 인덱스와 매핑됨
            if self.task == 'seg':
                images,labels = self.aug_apply(cv2.cvtColor(cv2.imread(self.dataset_path+img_file_name), cv2.COLOR_BGR2RGB),mask_or_box = mask_list,iter=self.iter)
            elif self.task == 'od':
                images,labels,classes  = self.aug_apply(cv2.cvtColor(cv2.imread(self.dataset_path+img_file_name), cv2.COLOR_BGR2RGB),mask_or_box = box_list, cls_name = cls_name,iter=self.iter)
            
            self.save_imgs(img_file_name.split(".")[0],images)
            self.coco_form_write(img_file_name.split(".")[0],labels,cat_list)
        
        with open(self.output_dir+"/aug_anno_"+str(self.task)+"_"+str(self.iter)+".json", "w") as handle:
            json.dump(self.form, handle)
            
    def aug_apply(self,image,mask_or_box:list,cls_name = [],iter=20):
        image_list = [image]
        label_list =[mask_or_box]
        cls_list = [cls_name]
        for i in range(iter):
            if self.task == "seg":
                augment = self.aug(image=image, masks = mask_or_box)
                image_list.append(augment['image'])
                label_list.append(augment['masks'] )
                
            elif self.task == "od":
                augment = self.aug(image=image, bboxes = mask_or_box, class_labels = cls_name)
                image_list.append(augment['image'])
                label_list.append(augment['bboxes'])
                cls_list.append(augment['class_labels'])
                  
        if self.task == "seg": return image_list, label_list
        elif self.task == "od": return image_list, label_list, cls_list
      
    def save_imgs(self,filenames,images):
        # save all augmented images to local (for single src image)
        if os.path.isdir(self.output_dir):pass
        else: os.mkdir(self.output_dir)
        for i in range(len(images)):
            cv2.imwrite(self.output_dir+filenames.split(".")[0]+"_aug_"+str(i)+".jpg",cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    
    def coco_form_write(self,filenames,labels,cat_list):
        if self.task == "seg":
            for count ,mask in enumerate(labels):
                self.image_id +=1
                self.form['images'].append({'date_captured': '2022',
                                        'file_name': self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",
                                        'id': self.image_id,
                                        'height': cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[0],
                                        'width': cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[1]})
            
                for cat in self.cats:
                    for id in range(len(cat_list)):
                        if cat["id"] == cat_list[id]:
                            if {'id':cat["id"],'name':cat["name"]} in self.form["categories"]:
                                pass
                            else:
                                self.form["categories"].append({'id':cat["id"],
                                                'name':cat["name"]})
                
                for i in range(len(mask)):
                    tmp = mask[i].copy()
                    #get contours of image
                    tmp = tmp.astype(np.uint8)
                    contours,hierachy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
                    
                    for cnt, cont in enumerate(contours):
                            segmentation = []
                            xmin,ymin,width,height = cv2.boundingRect(cont) #bounding box
                            if width * height < 3:
                                continue
                            image_height = tmp.shape[0]
                            image_width = tmp.shape[1]

                            cont = cont.flatten().tolist() #contour as 1d array has shape (x1,y1,x2,y2,...,x_n, y_n)
                            if len(cont) > 4: #only of at least 2 points are there
                                segmentation.append(cont)
                            else:
                                continue
                            if len(segmentation) == 0: #check again if segmentations are in list
                                continue
                            if (width * height) > 100 :
                                self.form["annotations"].append({'segmentation': segmentation,
                                                'area': width * height,
                                                    'image_id': self.image_id,
                                                    'iscrowd':0,
                                                    'bbox': [xmin,ymin,width,height],
                                                    "category_id": cat_list[i],
                                                    "id": self.all_annos})                            
                                self.all_annos += 1
        elif self.task == "od":
            for count ,boxes in enumerate(labels):
                self.image_id +=1
                self.form['images'].append({'date_captured': '2022',
                                        'file_name': self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",
                                        'id': self.image_id,
                                        'height': cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[0],
                                        'width': cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[1]})
            
                for cat in self.cats:
                    for id in range(len(cat_list)):
                        if cat["id"] == cat_list[id]:
                            if {'id':cat["id"],'name':cat["name"]} in self.form["categories"]:
                                pass
                            else:
                                self.form["categories"].append({'id':cat["id"],
                                                'name':cat["name"]})
                                
                for i in range(len(boxes)):                            
                    xmin,ymin,width,height = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3] #bounding box
                    if width * height < 3:
                        continue
                    image_height = cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[0]
                    image_width = cv2.imread(self.output_dir+filenames.split(".")[0]+"_aug_"+str(count)+".jpg",0).shape[1]

                    if (width * height) > 100 :
                        self.form["annotations"].append({'segmentation': [],
                                        'area': width * height,
                                            'image_id': self.image_id,
                                            'iscrowd':0,
                                            'bbox': [xmin,ymin,width,height],
                                            "category_id": cat_list[i],
                                            "id": self.all_annos})                            
                        self.all_annos += 1           

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_dataset_dir',help="image & json path from local filesystem", required=True)
    parser.add_argument('--output_dir',help="image & json path to export augmented image data & coco format json file" ,required=True)
    parser.add_argument('--labeling_type', type=str ,default="od", required=True)
    parser.add_argument('--iter',  type=int ,default=20)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    user_dir_path = args.user_dataset_dir
    output_dir = args.output_dir
    task_type = str(args.labeling_type)
    iter = args.iter
    
    augmentator(user_dir_path,output_dir,labeling_type = task_type,iter = iter)