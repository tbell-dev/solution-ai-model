# for after pred -> result -> post process -> data format change for sending front-end
import json
import numpy as np
import cv2
import pycocotools
from pycocotools.coco import COCO

from modules.post_proccessing import create_sub_mask_annotation

def create_coco_dict_seg(image,segmentations,bbox,id,idx):
    '''
    only creates coco dataset annotation field 
    '''
    json_data = {}
    json_data["annotations"] = []
    segmentation = []

    xmin,ymin,width,height = bbox.tolist()
    image_height = image.shape[0]
    image_width = image.shape[1]
    json_data["annotations"].append({'segmentation': segmentations,
                                    'area': width * height,
                                    'image_id': 0,
                                    'iscrowd':0,
                                    'bbox': [xmin,ymin,width,height],
                                    "category_id": id,
                                    "id": idx})
    
    return json_data

def create_coco_dict_od(bbox,id,idx):
    '''
    only creates coco dataset annotation field 
    '''
    json_data = {}
    json_data["annotations"] = []
    xmin,ymin,width,height = list(map(int,bbox.tolist()))
    json_data["annotations"].append({'id': idx,
                                     'image_id': 0,
                                     'category_id': id,
                                     'bbox':  list(map(int,bbox.tolist())),
                                     'area': width * height,
                                     'segmentation': [],
                                     'iscrowd':0
                                    })
    return json_data
    

def coco_format_inverter(result):
    coco_anno_type_json_list = []
    if "MASKS" in list(result.keys()):
        for i in range(len(result["MASKS"])):
            binary_mask = np.where(result["MASKS"][i] > 0,255,0)
            _, segmentations = create_sub_mask_annotation(binary_mask)
            json_data = create_coco_dict_seg(binary_mask,segmentations,result["BBOXES"][i],result["CLASSES"][i],i)
            coco_anno_type_json_list.append(json_data)
    # print(coco_anno_type_json_list)
    else:
        for i in range(len(result["bboxes__0"])):
            json_data = create_coco_dict_od(result["bboxes__0"][i],result["classes__1"][i],i)
            coco_anno_type_json_list.append(json_data)
    
    return coco_anno_type_json_list
        