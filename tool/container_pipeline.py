from genericpath import isdir
import torch, gc , os
import docker
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
import subprocess
from glob import glob
import os, argparse
import random, json 
from datetime import date
from pycocotools.coco import COCO

import argparse
import os
from typing import Dict, List, Tuple
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

# customization for force tensors to GPU from CPU##############################
from unittest.mock import patch
from functools import wraps

# for model contorl
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# for model convert to tensorflow
from tensorConverter.d2pb import D2toTensorflow

logger = setup_logger()
DEFAULT_MODEL_PATH = "/workspace/models/"

def patch_torch_stack(func):
    """
    Patch torch.stack to move its outputs to GPU
    """
    orig_stack= torch.stack

    def new_stack(*args, **kwargs):
        return orig_stack(*args, **kwargs).to('cuda')

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("torch.stack", side_effect=new_stack, wraps=orig_stack):
            return func(*args, **kwargs)
    return wrapper

@patch_torch_stack # added 0920
def export_tracing(format,torch_model, inputs,ouput_dir):
    global logger
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(ouput_dir, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, ouput_dir)
    elif format == "onnx":
        with PathManager.open(os.path.join(ouput_dir, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper

def get_sample_inputs_func(cfg,sample_image):

    if sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs
    
def model_export_to_ts(cfg,sample_image,ouput_dir):
    global logger    
    # logger = setup_logger()
    PathManager.mkdirs(ouput_dir)
    torch._C._jit_set_bailout_depth(1)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()
    
    # get sample data
    sample_inputs = get_sample_inputs_func(cfg,sample_image)
    # exported model has to be using tracing method
    exported_model = export_tracing("torchscript",torch_model, sample_inputs,ouput_dir)
        
    logger.info("Success.")

def model_export_to_onnx(cfg,sample_image,ouput_dir):
    global logger
    # logger = setup_logger()
    PathManager.mkdirs(ouput_dir)
    torch._C._jit_set_bailout_depth(1)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()
    
    # get sample data
    sample_inputs = get_sample_inputs_func(cfg,sample_image)

    # convert and save model
    exported_model = export_tracing("onnx",torch_model, sample_inputs,ouput_dir)
        
    logger.info("Success.")

def get_loss(metric_path):
    metric_path = os.path.join(metric_path,"metrics.json")
    f = open(metric_path, 'r')
    lines = f.readlines()
    json_data = []
    for line in lines:
        json_ins = json.loads(line)
        json_data.append(json_ins)

    return json_data[-1]

def model_validation(cfg,current_prj_name):
    model_repo = "/".join(cfg.OUTPUT_DIR.split("/")[:-1])
    prjs = [i for i in glob(model_repo+"/*") if current_prj_name in i.split("/")[-1]]
    if len(prjs) == 1 : 
        return prjs[0]+"/model_final.pth"
    elif len(prjs) == 0 :
        return model_repo+"/model_final.pth"
    else:
        latest_result = prjs[-1]
        previous_result = prjs[:-1]
        latest_loss = get_loss(latest_result)["total_loss"]
        
        previous_losses = []
        for i in range(len(previous_result)):
            previous_losses.append(get_loss(previous_result[i])["total_loss"])
                
        if min(previous_losses) < latest_loss : 
            min_loss = min(previous_losses)
            idx = previous_losses.index(min_loss)
            return previous_result[idx]+"/model_final.pth"
        
        elif min(previous_losses) >= latest_loss : 
            return latest_result+"/model_final.pth"

def model_version_val(cfg):
    prj_repo = "/".join(cfg.OUTPUT_DIR.split("/")[:-1])
    versions = [i for i in glob(prj_repo+"/*")]
    if len(versions) == 1:
        return versions[0]+"/model_final.pth"
    else:
        latest_result = versions[-1]
        previous_result = versions[:-1]
        latest_loss = get_loss(latest_result)["total_loss"]
        previous_losses = []
        
        for i in range(len(previous_result)):
            previous_losses.append(get_loss(previous_result[i])["total_loss"])
        
        if min(previous_losses) < latest_loss : 
            min_loss = min(previous_losses)
            idx = previous_losses.index(min_loss)
            return previous_result[idx]+"/model_final.pth"
        
        elif min(previous_losses) >= latest_loss : 
            return latest_result+"/model_final.pth"
        
def get_weight_path(cfg):
    prj_repo = "/".join(cfg.OUTPUT_DIR.split("/")[:-1])
    versions = [i for i in glob(prj_repo+"/*")]
    if len(versions) == 1:
        return versions[0]+"/model_final.pth"
    else:
        latest_result = versions[-1]
        previous_result = versions[:-1]
        return latest_result+"/model_final.pth"

def coco_style_gen():
    today = date.today()
    files = {}
    files['info'] = {"year": str(today.year), "version": "1.0", "description": "Data format for active learning & auto labeling",
                     "date_created": str(today.month)+"/"+str(today.day)}
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
    def __init__(self,augmentated_dir,project_name,model_name,epoch,learning_rate,batch_size,labeling_type = "bbox",v=0.7,op="cp"):
        self.output_dir = augmentated_dir
        self.v = v
        self.project_name = str(project_name)
        self.model_name = model_name
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cfg = None
        self.labeling_type = labeling_type
        self.train_path = self.output_dir+"/train/"
        self.val_path = self.output_dir+"/val/"
        
        self.cfg = get_cfg()
        self.cats = spliter(self.output_dir,v=self.v,op=op)
        self.register_data()
        
    def register_data(self):
        self.data_clear()
        register_coco_instances("my_dataset_train", {}, self.train_path+"train.json", self.train_path)
        register_coco_instances("my_dataset_val", {}, self.val_path+"val.json",self.val_path)
        
    def data_clear(self):
        DatasetCatalog.clear()
        
    def start(self):
        if self.labeling_type == 'bbox':
            model_base = "COCO-Detection/faster_rcnn_"
            task_type = "od"
        elif self.labeling_type == 'polygon':
             model_base = "COCO-InstanceSegmentation/mask_rcnn_"
             task_type = "seg"

        model_pth = model_base + self.model_name+".yaml"

        self.cfg.merge_from_file(model_zoo.get_config_file(model_pth))
        self.cfg.OUTPUT_DIR = f"/workspace/output/{task_type}/{self.project_name}"
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_pth)  # Let training initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = self.learning_rate  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = self.epoch    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.cats)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        
        if os.path.isdir(self.cfg.OUTPUT_DIR):
            # model_repo = "/workspace/output"
            # same_prjs = [i for i in glob(model_repo+'/*') if self.project_name in i.split("/")[-1]]
            # prj_dir = f"/workspace/output/{task_type}/{self.project_name}"
            versions = [i for i in glob(self.cfg.OUTPUT_DIR+"/*")]
            self.cfg.OUTPUT_DIR = f"/workspace/output/{task_type}/{self.project_name}/{str(len(versions)+1)}"
            os.makedirs(self.cfg.OUTPUT_DIR)
            # Open file in write mode
            with open(self.cfg.OUTPUT_DIR+'/model_config_name.txt', 'w') as f:
                # Write content to file
                f.write(f'{self.model_name}\n')
        else:
            self.cfg.OUTPUT_DIR = self.cfg.OUTPUT_DIR+"/1"
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            # Open file in write mode
            with open(self.cfg.OUTPUT_DIR+'/model_config_name.txt', 'w') as f:
                # Write content to file
                f.write(f'{self.model_name}\n')
            
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        
        # detectron2 -> torch
        torch.save(trainer.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, "model_final_ckpt.pth"))
        model = build_model(self.cfg)
        model.load_state_dict(torch.load(os.path.join(self.cfg.OUTPUT_DIR, "model_final_ckpt.pth")))
        model.eval()
        torch.save(model,self.cfg.OUTPUT_DIR+'/model_final_torch.pt')
        
        #for free gpu
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.cfg

def configure_model_dir(task_type,cfg,sample_image,output_dir_name): 
    deploy_dir = ""
    if os.path.isdir(DEFAULT_MODEL_PATH+"/"+output_dir_name):
        deploy_dir = DEFAULT_MODEL_PATH+"/"+output_dir_name+"/1"
        os.mkdir(deploy_dir)
    else: 
        os.makedirs(DEFAULT_MODEL_PATH+"/"+output_dir_name+"/1")
        deploy_dir = DEFAULT_MODEL_PATH+"/"+output_dir_name+"/1"
    
    if task_type == "seg":
        shutil.copy(DEFAULT_MODEL_PATH+"mask_rcnn/config.pbtxt",DEFAULT_MODEL_PATH+output_dir_name+"/config.pbtxt")
        shutil.copytree(DEFAULT_MODEL_PATH+"infer_pipeline",DEFAULT_MODEL_PATH+"infer_pipeline_"+output_dir_name)
        with open(DEFAULT_MODEL_PATH+"infer_pipeline_"+output_dir_name+'/config.pbtxt') as f:
            txt = f.read()
        
        new_txt = txt.replace('mask_rcnn',output_dir_name)
        with open(DEFAULT_MODEL_PATH+"infer_pipeline_"+output_dir_name+'/config.pbtxt',"w") as f:
            f.write(new_txt)
                
    elif task_type == "od":
        shutil.copy(DEFAULT_MODEL_PATH+"faster_rcnn/config.pbtxt",DEFAULT_MODEL_PATH+output_dir_name+"/config.pbtxt" )
        
    model_export_to_ts(cfg,sample_image,deploy_dir) 

def deploy_servable_model(cfg,sample_image,output_dir_name): # output_dir_name = [task,project_name,version]
    deploy_dir = ""
    if os.path.isdir(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/"+output_dir_name[1]):
        deploy_dir = DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/"+output_dir_name[1]+"/"+output_dir_name[2]
        os.mkdir(deploy_dir)
    else :
        deploy_dir = DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/"+output_dir_name[1]+"/1"
        os.makedirs(deploy_dir)
        
    if output_dir_name[0] == "seg":
        shutil.copy(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/mask_rcnn/config.pbtxt",
                    DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/"+output_dir_name[1]+"/config.pbtxt")
        
        if os.path.isdir(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]):
            
            os.mkdir(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]+"/"+output_dir_name[2])
            with open(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]+'/config.pbtxt') as f:
                txt = f.read()
            
            versions = [i.split("/")[-1] for i in glob(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]+"/*") 
                        if "." not in i.split("/")[-1]]
            versions.remove(output_dir_name[2]) 
            last_version = max(list(map(int, versions)))

            text =  txt.replace(f'      model_name: "{output_dir_name[1]}"\n      model_version: {str(last_version)}',
                                f'      model_name: "{output_dir_name[1]}"\n      model_version: {output_dir_name[2]}')
            
        else:
            shutil.copytree(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline",
                            DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1])
            with open(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]+'/config.pbtxt') as f:
                txt = f.read()
                
            text =  txt.replace(f'      model_name: "mask_rcnn"\n      model_version: 1',
                                f'      model_name: "{output_dir_name[1]}"\n      model_version: {output_dir_name[2]}')
        
        with open(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/infer_pipeline_"+output_dir_name[1]+'/config.pbtxt',"w") as f:
            f.write(text)
            
    elif output_dir_name[0] == "od":
        shutil.copy(DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/faster_rcnn/config.pbtxt",
                    DEFAULT_MODEL_PATH+"/"+output_dir_name[0]+"/"+output_dir_name[1]+"/config.pbtxt")
    # detectron2 -> torchscript : export torchscript to models(servable)    
    model_export_to_ts(cfg,sample_image,deploy_dir)
    # detectron2 -> onnx : export onnx to output(trained)
    model_export_to_onnx(cfg,sample_image,cfg.OUTPUT_DIR)
    return output_dir_name[1]

def model_ctl(ctl,model_name,host= "172.17.0.1",port = 8000):
    triton_client = httpclient.InferenceServerClient(url=host+":"+str(port), verbose=False)
    if ctl == "load":
        triton_client.load_model(model_name)
    if ctl == "unload":
        triton_client.unload_model(model_name)
    return [stats for stats in triton_client.get_model_repository_index() if stats["name"] == model_name][0]
         
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',help="image & json path from local filesystem, root of train/ & val/", type=str ,default="/workspace/dataset/")
    parser.add_argument('--project_name', type=str ,default="0",required=True)
    parser.add_argument('--labeling_type', type=str ,default="bbox", required=True)
    parser.add_argument('--serving_host', type=str ,default="172.17.0.1")
    parser.add_argument('--model_name', type=str ,default="R_50_FPN_3x")
    parser.add_argument('--epoch', type=int ,default=1000)
    parser.add_argument('--lr', type=float ,default=0.00025)
    parser.add_argument('--batch_size', type=int ,default=2)
    parser.add_argument('--split',  type=float ,default=0.7)
    # parser.add_argument('--outputdir',  type=str ,default="/workspace/models/") # mounted with '{project_root}/models' on host
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_dir = args.dataset_dir
    labeling_type = args.labeling_type
    project_name = args.project_name
    v = args.split
    serving_host = args.serving_host

    model_name = args.model_name
    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size

    task_type = ""
    
    train = trainer(augmentated_dir = dataset_dir,
                    project_name = project_name,
                    labeling_type = labeling_type,
                    model_name = model_name,
                    epoch = epoch,
                    learning_rate = lr,
                    batch_size = batch_size,
                    v=0.8,
                    op="cp")
    cfg = train.start()
    
    # weight_path = model_validation(cfg,project_name)
    # weight_path = model_version_val(cfg)
    weight_path = get_weight_path(cfg)
    
    
    # if "/".join(weight_path.split("/")[:-1]) == cfg.OUTPUT_DIR:
    cfg.MODEL.WEIGHTS = weight_path
    sample_image = [i for i in glob(dataset_dir+"/val/*.jpg")][0]
    model_name = deploy_servable_model(cfg,sample_image,output_dir_name=cfg.OUTPUT_DIR.split("/")[-3:])
        # D2toTensorflow(cfg,weight_path,dataset_dir,cfg.OUTPUT_DIR+"/model")
        # if labeling_type == "bbox":
        #     model_ctl("load",model_name,host = str(serving_host),port = 8000)
        # elif labeling_type == "polygon" or labeling_type == "segment":
        #     model_ctl("load",model_name,host = str(serving_host),port = 8001)
        #     model_ctl("load","infer_pipeline_"+model_name,host =  str(serving_host),port = 8001)
        
        # change directory permission
        # cmds = ["chmod -R 664 /workspace/dataset","chmod 775 /workspace/dataset", "chown -R tbelldev:tbelldev /workspace/dataset",
        #         "chmod 775 /workspace/dataset/train", 
        #         "chmod 775 /workspace/dataset/val", 
        #         "chmod -R 664 /workspace/output","chmod 775 /workspace/output", "chown -R tbelldev:tbelldev /workspace/output",
        #         "chmod -R 664 /workspace/models","chmod 775 /workspace/models", "chown -R tbelldev:tbelldev /workspace/models"]
    cmds = ["chown -R tbelldev:tbelldev /workspace/dataset",
            "chown -R tbelldev:tbelldev /workspace/output",
            "chown -R tbelldev:tbelldev /workspace/models"]
    for cmd in cmds:
        subprocess.call(cmd, shell=True, stdin=subprocess.PIPE,
                universal_newlines=True)
        
        