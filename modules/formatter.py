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

# def create_coco_dict_seg(image,segmentations,bbox,id,idx):
#     '''
#     only creates coco dataset annotation field 
#     '''
#     json_data = {}
#     json_data["annotations"] = []
#     segmentation = []

#     xmin,ymin,width,height = bbox.tolist()
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     json_data["annotations"].append({'segmentation': segmentations,
#                                     'area': width * height,
#                                     'image_id': 0,
#                                     'iscrowd':0,
#                                     'bbox': [xmin,ymin,width,height],
#                                     "category_id": id,
#                                     "id": idx})
    
#     return json_data

def create_coco_dict_seg_v2(image,mask,bbox,id,idx,score,image_id = 0):
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
    contours,_ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
    # print(len(contours))
    for cnt, cont in enumerate(contours):
            segmentation = []
            xmin,ymin,width,height = cv2.boundingRect(cont) #bounding box
            if width * height < 3:
                continue
            cont = cont.flatten().tolist() #contour as 1d array has shape (x1,y1,x2,y2,...,x_n, y_n)
            if len(cont) > 4: #only of at least 2 points are there
                segmentation.append(cont)
            else:
                continue
            if len(segmentation) == 0: #check again if segmentations are in list
                continue
            if (width * height) > 100 :
                data = {
                        'segmentation': segmentation,
                        'area': width * height,
                        'image_id': image_id,
                        'iscrowd':0,
                        'bbox': [xmin,ymin,width,height],
                        "category_id": id,
                        "id": idx,
                        'score':score,
                        "keypoints":[],
                        "num_keypoints":0
                        }
    
    return data

def create_coco_dict_od(bbox,id,idx,score,image_id = 0):
    '''
    only creates coco dataset annotation field 
    '''
    xmin,ymin,width,height = list(map(int,bbox.tolist()))
    # result_list.append({'id': idx,
    #                     'image_id': 0,
    #                     'category_id': id,
    #                     'bbox':  list(map(int,bbox.tolist())),
    #                     'area': width * height,
    #                     'segmentation': [],
    #                     'iscrowd':0
    #                     })
    data = {'id': idx,
            'image_id': image_id,
            'category_id': id,
            'bbox':  list(map(int,bbox.tolist())),
            'area': width * height,
            'segmentation': [],
            'iscrowd':0,
            'score':score,
            "keypoints":[],
            "num_keypoints":0
            }
    return data

def create_coco_dict_seg_v2_batch(image,mask,bbox,id,idx,score,image_id = 0):
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
    contours,_ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
    # print(len(contours))
    data = None
    for cnt, cont in enumerate(contours):
            segmentation = []
            xmin,ymin,width,height = cv2.boundingRect(cont) #bounding box
            if width * height < 3:
                continue
            cont = cont.flatten().tolist() #contour as 1d array has shape (x1,y1,x2,y2,...,x_n, y_n)
            if len(cont) > 4: #only of at least 2 points are there
                segmentation.append(cont)
            else:
                continue
            if len(segmentation) == 0: #check again if segmentations are in list
                continue
            if (width * height) > 100 :
                data = {
                        'segmentation': segmentation,
                        'area': width * height,
                        'image_id': image_id,
                        'iscrowd':0,
                        'bbox': [xmin,ymin,width,height],
                        "category_id": id,
                        "id": idx,
                        "keypoints":[],
                        "num_keypoints":0,
                        "score":score
                        }
    return data

def create_coco_dict_od_batch(bbox,id,idx,score,image_id = 0):
    '''
    only creates coco dataset annotation field 
    '''
    xmin,ymin,width,height = list(map(int,bbox.tolist()))
    # result_list.append({'id': idx,
    #                     'image_id': 0,
    #                     'category_id': id,
    #                     'bbox':  list(map(int,bbox.tolist())),
    #                     'area': width * height,
    #                     'segmentation': [],
    #                     'iscrowd':0
    #                     })
    data = {'id': idx,
            'image_id': image_id,
            'category_id': id,
            'bbox':  list(map(int,bbox.tolist())),
            'area': width * height,
            'segmentation': [],
            'iscrowd':0,
            "keypoints":[],
            "num_keypoints":0,
            'score':score
            }
    return data
    

def coco_format_inverter(result):
    json_data = {}
    json_data["annotations"] = []
    if "MASKS" in list(result.keys()):
        for i in range(len(result["MASKS"])):
            binary_mask = np.where(result["MASKS"][i] > 0,255,0)
            data = create_coco_dict_seg_v2(binary_mask,result["MASKS"][i],result["BBOXES"][i],result["CLASSES"][i],i,result["SCORES"][i])
            json_data["annotations"].append(data)
    else:
        for i in range(len(result["bboxes__0"])):
            data = create_coco_dict_od(result["bboxes__0"][i],result["classes__1"][i],i,result["scores__2"][i])
            json_data["annotations"].append(data)
    return json_data

def coco_format_inverter_batch(result_list,image_list,is_local=False):
    json_data = {}
    json_data["annotations"] = []
    print("image_list from formater: ",image_list)
    for r in range(len(result_list)):
        if is_local:
            file_name = str(image_list[r]).split(" ")[1].replace("'","").split(".")[0]
        else:
            file_name = str(image_list[r]).split("/")[-1].split(".")[0]
        if "MASKS" in list(result_list[r].keys()):
            for i in range(len(result_list[r]["MASKS"])):
                binary_mask = np.where(result_list[r]["MASKS"][i] > 0,255,0)
                data = create_coco_dict_seg_v2_batch(binary_mask,
                                               result_list[r]["MASKS"][i],
                                               result_list[r]["BBOXES"][i],
                                               result_list[r]["CLASSES"][i],
                                               i,
                                               result_list[r]["SCORES"][i],
                                               image_id = str(file_name))
                if data == None: pass
                else:json_data["annotations"].append(data)
        else:
            for i in range(len(result_list[r]["bboxes__0"])):
                data = create_coco_dict_od_batch(result_list[r]["bboxes__0"][i],
                                           result_list[r]["classes__1"][i],
                                           i,
                                           result_list[r]["scores__2"][i],
                                           image_id = str(file_name))
                json_data["annotations"].append(data)
    
    
    for i in range(len(json_data["annotations"])):
        json_data["annotations"][i]["id"] = i
                
    return json_data
        