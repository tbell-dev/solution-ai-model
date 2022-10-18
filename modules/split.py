from glob import glob
import random
import json
from modules.formatter import coco_style_gen
from pycocotools.coco import COCO
import shutil , os

def save_split_coco(augmented_dir,cocodata,data_type = 'train'):
    if data_type == 'train':
        file_path = augmented_dir + "/train/train.json"
    else : file_path = augmented_dir + "/val/val.json"
    
    with open(file_path, "w") as handle:
        json.dump(cocodata, handle)

def spliter(augmented_dir,v = 0.7):
    imgs = [i for i in glob(augmented_dir+"/*.jpg")]
    anno = [i for i in glob(augmented_dir+"/*.json")][0]

    random.shuffle(imgs)
    train,val = imgs[:int(len(imgs)*v)],imgs[int(len(imgs)*v):]

    if os.path.isdir(augmented_dir+"/train") : pass
    else : os.mkdir(augmented_dir+"/train")
    if os.path.isdir(augmented_dir+"/val") : pass
    else : os.mkdir(augmented_dir+"/val")
    for c,data in enumerate([train,val]):
        for d in data:
            if c == 0:
                shutil.move(d,augmented_dir+"/train/"+d.split("/")[-1])
            elif c == 1:
                shutil.move(d,augmented_dir+"/val/"+d.split("/")[-1])
                
    ###############################################################################################

    # get coco file
    coco_annotation = COCO(anno)

    # get category field from origin
    train_coco, val_coco = [coco_style_gen() for i in range(2)]
    train_coco["categories"] , val_coco["categories"] = [coco_annotation.loadCats(coco_annotation.getCatIds()) for i in range(2)]

    # extract annos coresponding train imgs from coco file
    imgids = coco_annotation.getImgIds()
    for i in range(len(imgids)):
        ann_ids = coco_annotation.getAnnIds(imgIds=[imgids[i]], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        img_info = coco_annotation.loadImgs([imgids[i]])[0]
        if img_info["file_name"] in train: 
            newfile_name = '/'.join(img_info["file_name"].split("/")[:-1])+"/train/"+img_info["file_name"].split("/")[-1]
            img_info["file_name"] = newfile_name
            train_coco["images"].append(img_info)
            for ann in anns:
                train_coco["annotations"].append(ann)
            continue
        
        elif img_info["file_name"] in val:
            newfile_name = '/'.join(img_info["file_name"].split("/")[:-1])+"/val/"+img_info["file_name"].split("/")[-1]
            img_info["file_name"] = newfile_name
            val_coco["images"].append(img_info)
            for ann in anns:
                val_coco["annotations"].append(ann)
            continue
        
    save_split_coco(augmented_dir,train_coco,data_type = "train")
    save_split_coco(augmented_dir,val_coco,data_type = "val")
    os.remove(anno)
    return train_coco["categories"]