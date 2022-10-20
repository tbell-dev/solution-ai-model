import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2

from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os

from matplotlib import pyplot as plt




default_class ={0:"person"}

def get_boxes(outputs:list):
    boxes = []
    for output in outputs:
        # output["instances"].pred_boxes.tensor.cpu().numpy()
        for pred_mask in output['instances'].pred_boxes.tensor.cpu():
            boxes.append(np.concatenate(pred_mask.numpy().tolist()))
    return boxes

def get_mask_from_boolen(outputs:list):
    imsize = output['instances'].image_size #(height,width)
    for output in outputs:
        for pred_mask in output['instances'].pred_masks.cpu():
            mask = pred_mask.numpy().astype('uint8')
            resmask = mask.reshape(imsize[0],imsize[1],1)
            transe = np.where(resmask > 0,255,0)
    return transe

def get_polygon_from_mask(outputs:list):
    
    poly_group=[]    
    for output in outputs:
        ploygons = []
        for pred_mask in output['instances'].pred_masks.cpu():
            # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
            # Convert it to a 8-bit numpy array, which can then be used to find contours
            mask = pred_mask.numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            ploygons.append(contour[0])
        poly_group.append(ploygons)

    # for ploygon in ploygons:
    #     cv2.drawContours(imco, [ploygon], -1, (0,255,0), 1)
    
    return poly_group

def inference_class_filter(output,class_name:list):
    pred = []
    for i in class_name:
        if i in default_class.values():
            for j in range(len(default_class)):
                if default_class.get(j) == i:
                    pred.append(output["instances"][j])
        else:pred = None
    return pred
#######################################################################
def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)
        
        if type(poly) == MultiPolygon:
            poly = list(poly)
            for i in range(len(poly)):
                segmentations.append(np.array(poly[i].exterior.coords).ravel().tolist())
        else:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def show_result_from_local_inference(impath,id_list,cat_list,predictor,task_type):
    image = cv2.imread(impath)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    outputs = predictor(img)
    imco = img.copy()
    contours = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,0,0)
    if task_type == "od":
        for i,box in enumerate(outputs['instances'].pred_boxes):
            text = ""
            x1,y1,x2,y2 = box.cpu().numpy()
            if str(outputs['instances'].pred_classes[i].cpu().numpy()) == "0": 
                text = str(cat_list[0])+" "+ str(round(float(outputs['instances'].scores[i].cpu().numpy()),2))
                color = (255,0,0)
            elif str(outputs['instances'].pred_classes[i].cpu().numpy()) == "1": 
                text = str(cat_list[0])+" "+str(round(float(outputs['instances'].scores[i].cpu().numpy()),2))
                color = (0,255,0)
            elif str(outputs['instances'].pred_classes[i].cpu().numpy()) == "2": 
                text = str(cat_list[0])+" "+str(round(float(outputs['instances'].scores[i].cpu().numpy()),2))
                color = (0,0,255)
                
            cv2.rectangle(imco,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            cv2.putText(imco,text,(int(x1),int(y1-10)),font,0.5,color,2)
            print(text)
            
    elif task_type == "seg":
        for pred_mask in outputs['instances'].pred_masks.cpu():
            # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
            # Convert it to a 8-bit numpy array, which can then be used to find contours
            mask = pred_mask.numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            contours.append(contour[0]) # contour is a tuple (OpenCV 4.5.2), so take the first element which is the array of contour points

        for contour in contours:
            cv2.drawContours(imco, [contour], -1, (255,0,0), 2)
        
    plt.imshow(imco)
    plt.show()
