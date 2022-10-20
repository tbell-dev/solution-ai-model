import torch, gc
from modules.split import spliter
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
        cfg.OUTPUT_DIR = self.output_dir +"/"+self.project_name+"/"
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

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        #for free gpu
        gc.collect()
        torch.cuda.empty_cache()
        
        return cfg
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',help="image & json path from local filesystem, root of train/ & val/", required=True)
    parser.add_argument('--project_name', type=str ,default="0", required=True)
    parser.add_argument('--labeling_type', type=str ,default="od", required=True)
    parser.add_argument('--split',  type=float ,default=0.7)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_dir = args.dataset_dir
    labeling_type = args.labeling_type
    project_name = args.project_name
    v = args.split
    
    train = trainer(dataset_dir,project_name,labeling_type,v=0.7,op="cp")
    cfg = train.start()