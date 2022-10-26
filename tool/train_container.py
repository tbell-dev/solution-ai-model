from genericpath import isdir
import torch, gc
# from modules.split import spliter
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
import shutil
from glob import glob
import os, argparse
import random, json 
from datetime import date
from pycocotools.coco import COCO

def coco_style_gen():
    today = date.today()
    files = {}
    files['info'] = {"year": str(today.year), "version": "1.0", "description": "Person Segmentation", "date_created": str(today.month)+"/"+str(today.day)}
    files['licenses'] = [{'id': 1,
        'name': 'TBell - sslo general license v1',
        'url': 'https://sslo.ai/'}]
    files["type"] = "instances"
    files['categories'] = []
    files["annotations"] = []
    files['images'] = []
    return files

def save_split_coco(augmented_dir,cocodata,data_type = 'train'):
    if data_type == 'train':
        file_path = augmented_dir + "/train/train.json"
    else : file_path = augmented_dir + "/val/val.json"
    
    with open(file_path, "w") as handle:
        json.dump(cocodata, handle)

def spliter(augmented_dir,v = 0.7, op = "cp"):
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
                if op == "cp":shutil.copy(d,augmented_dir+"/train/"+d.split("/")[-1])
                else:shutil.move(d,augmented_dir+"/train/"+d.split("/")[-1])
            elif c == 1:
                if op == "cp":shutil.copy(d,augmented_dir+"/val/"+d.split("/")[-1])
                else:shutil.move(d,augmented_dir+"/val/"+d.split("/")[-1])
                
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
        # change file_name fields cause in container dir path is different.
        img_info["file_name"] = "/workspace/dataset/"+img_info["file_name"].split("/")[-1]
        
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
    
    if op == "cp": pass
    else: os.remove(anno)
    
    return train_coco["categories"]

class trainer:
    def __init__(self,augmentated_dir,project_name,labeling_type = "seg",v=0.7,op="cp"):
        self.output_dir = augmentated_dir
        self.v = v
        self.project_name = str(project_name)
        self.cfg = None
        self.task_type = labeling_type
        self.train_path = self.output_dir+"/train/"
        self.val_path = self.output_dir+"/val/"
        self.cats = spliter(self.output_dir,v=self.v,op=op)
        self.register_data()
        
    def register_data(self):
        self.data_clear()
        register_coco_instances("my_dataset_train", {}, self.train_path+"train.json", self.train_path)
        register_coco_instances("my_dataset_val", {}, self.val_path+"val.json",self.val_path)
        
    def data_clear(self):
        DatasetCatalog.clear()
        
    def start(self):
        cfg = get_cfg()
        
        model_pth = ""
        if self.task_type == 'bbox':
            model_pth = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        elif self.task_type == 'seg':
             model_pth = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
             
        cfg.merge_from_file(model_zoo.get_config_file(model_pth))
        cfg.OUTPUT_DIR = "/workspace/output/"+self.project_name
        cfg.DATASETS.TRAIN = ("my_dataset_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_pth)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.cats)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        
        self.cfg = cfg
        
        if os.path.isdir(self.cfg.OUTPUT_DIR):
            model_repo = "/workspace/output"
            same_prjs = [i for i in glob(model_repo+'/*') if self.project_name in i.split("/")[-1]]
            self.cfg.OUTPUT_DIR = "/workspace/output/"+self.project_name+"_"+str(len(same_prjs))
            os.makedirs(self.cfg.OUTPUT_DIR)
        else: os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        #for free gpu
        gc.collect()
        torch.cuda.empty_cache()
        
        return cfg
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',help="image & json path from local filesystem, root of train/ & val/", type=str ,default="/workspace/dataset/")
    parser.add_argument('--project_name', type=str ,default="0",required=True)
    parser.add_argument('--labeling_type', type=str ,default="bbox", required=True)
    parser.add_argument('--split',  type=float ,default=0.7)
    parser.add_argument('--ouput_host',help="root model repo output on host", type=str,required=True,default="/home/tbelldev/workspace/autoLabeling/api_test/model_repo/")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_dir = args.dataset_dir
    labeling_type = args.labeling_type
    project_name = args.project_name
    v = args.split
    host_path = args.ouput_host
    file_name_prefix = ""
    train = trainer(dataset_dir,project_name,labeling_type,v=0.7,op="cp")
    cfg = train.start()
    
    cfg.MODEL.WEIGHTS = host_path+cfg.OUTPUT_DIR.split("/")[-1]+"/model_final.pth"
    if labeling_type == "bbox" : file_name_prefix = "faster_rcnn_R_101_FPN_3x"
    if labeling_type == "polygon" : file_name_prefix = "mask_rcnn_R_101_FPN_3x"
    with open(cfg.OUTPUT_DIR+"/"+file_name_prefix+"_"+project_name+".yaml", "w") as f: 
        
        cfg.OUTPUT_DIR = host_path+cfg.OUTPUT_DIR.split("/")[-1]
        f.write(cfg.dump())
    