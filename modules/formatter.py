# for after pred -> result -> post process -> data format change for sending front-end
import json
import numpy as np
import cv2
import pycocotools
from pycocotools.coco import COCO

from modules.post_proccessing import create_sub_mask_annotation
from datetime import date

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

def create_coco_dict_seg_v2(image,mask,bbox,id,idx):
    '''
    only creates coco dataset annotation field 
    '''
    json_data = {}
    json_data["annotations"] = []
    segmentation = []

    xmin,ymin,width,height = bbox.tolist()
    image_height = image.shape[0]
    image_width = image.shape[1]
    tmp = mask.copy()
    #get contours of image
    tmp = tmp.astype(np.uint8)
    contours,hierachy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
    # print(len(contours))
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
                json_data["annotations"].append({'segmentation': segmentation,
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
            # _, segmentations = create_sub_mask_annotation(binary_mask)
            # json_data = create_coco_dict_seg(binary_mask,segmentations,result["BBOXES"][i],result["CLASSES"][i],i)
            json_data = create_coco_dict_seg_v2(binary_mask,result["MASKS"][i],result["BBOXES"][i],result["CLASSES"][i],i)
            coco_anno_type_json_list.append(json_data)
            # coco_anno_type_json_list.append(json_data)
    # print(coco_anno_type_json_list)
    else:
        for i in range(len(result["bboxes__0"])):
            json_data = create_coco_dict_od(result["bboxes__0"][i],result["classes__1"][i],i)
            coco_anno_type_json_list.append(json_data)
    
    return coco_anno_type_json_list
        