import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
from matplotlib import pyplot as plt

#custom helper module
from modules.post_proccessing import *

# make data format for REST api
from modules.formatter import *

# triton serving client
from modules.triton_serving import *


def get_base_model(label_type:str):
    
    cfg = get_cfg()
    
    if label_type == "bbox":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    elif label_type == "polygon":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif label_type == "segment":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
    return cfg       

def inference_local(image,class_name:list,label_type = "bbox"):
    
    cfg = get_base_model(label_type)
    
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    
    filtered_outputs = inference_class_filter(outputs,class_name) # filtered_outputs : list (ex. [Instance(),Instance(),Instance()])
    
    if label_type == "bbox":
        boxes = get_boxes(filtered_outputs)
        return boxes
    
    elif label_type == "polygon":
        polygon = get_polygon_from_mask(filtered_outputs)
        return polygon
    
    elif label_type == "segment":
        binary_mask = get_mask_from_boolen(filtered_outputs)
        return binary_mask
    
def inference_triton(image,class_name:list,port=8000,label_type = "bbox"):
    # inference using triton serving container , with httpclient 
    confidence = 0.60
    
    if label_type == "bbox":
        task_type = "od"
        model_name = "faster_rcnn"
        pred = inference(model_name,image_path,task_type,port=port)
        result = infer_result_filter(pred,task_type,confidence,"person")
        
    elif label_type == "polygon" or label_type == "segment":
        task_type = "seg"
        model_name = "infer_pipeline"
        pred = inference(model_name,image_path,task_type,port=port)
        result = infer_result_filter(pred,task_type,confidence,"person")
                
    # elif label_type == "segment":
    #     task_type = "seg"
    #     model_name = "infer_pipeline"
    #     pred = inference(model_name,image_path,task_type)
    #     result = infer_result_filter(pred,task_type,confidence,"person")
    
    response_data = coco_format_inverter(result)
    
    return response_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help="image path from local filesystem", required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', type=str ,default="od", required=True)
    parser.add_argument('--conf', type=float , default=0.6)
    parser.add_argument('--class_name', type=str ,default="person")
    parser.add_argument('--serving-port', type=int ,default=8000)
    parser.add_argument('--serving-host', type=str ,default="localhost")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    image_path = args.image
    model_name = args.model
    task_type = str(args.task)
    confidence = args.conf
    class_name =  args.class_name
    port = int(args.serving_port)
    host = str(args.serving_host)
    
    pred = inference(model_name,image_path,task_type,port=port,host=host)
    result = infer_result_filter(pred,task_type,confidence,class_name)
    response_data = coco_format_inverter(result)
    
    print(response_data)
