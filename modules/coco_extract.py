import json
from pycocotools.coco import COCO
import requests
import cv2 , numpy as np
from PIL import Image
from modules.formatter import coco_style_gen
def coco_cats():
    categories= [
    { "supercategory": "person", "id": 1, "name": "person" },
    { "supercategory": "vehicle", "id": 2, "name": "bicycle" },
    { "supercategory": "vehicle", "id": 3, "name": "car" },
    { "supercategory": "vehicle", "id": 4, "name": "motorcycle" },
    { "supercategory": "vehicle", "id": 5, "name": "airplane" },
    { "supercategory": "vehicle", "id": 6, "name": "bus" },
    { "supercategory": "vehicle", "id": 7, "name": "train" },
    { "supercategory": "vehicle", "id": 8, "name": "truck" },
    { "supercategory": "vehicle", "id": 9, "name": "boat" },
    { "supercategory": "outdoor", "id": 10, "name": "traffic light" },
    { "supercategory": "outdoor", "id": 11, "name": "fire hydrant" },
    { "supercategory": "outdoor", "id": 13, "name": "stop sign" },
    { "supercategory": "outdoor", "id": 14, "name": "parking meter" },
    { "supercategory": "outdoor", "id": 15, "name": "bench" },
    { "supercategory": "animal", "id": 16, "name": "bird" },
    { "supercategory": "animal", "id": 17, "name": "cat" },
    { "supercategory": "animal", "id": 18, "name": "dog" },
    { "supercategory": "animal", "id": 19, "name": "horse" },
    { "supercategory": "animal", "id": 20, "name": "sheep" },
    { "supercategory": "animal", "id": 21, "name": "cow" },
    { "supercategory": "animal", "id": 22, "name": "elephant" },
    { "supercategory": "animal", "id": 23, "name": "bear" },
    { "supercategory": "animal", "id": 24, "name": "zebra" },
    { "supercategory": "animal", "id": 25, "name": "giraffe" },
    { "supercategory": "accessory", "id": 27, "name": "backpack" },
    { "supercategory": "accessory", "id": 28, "name": "umbrella" },
    { "supercategory": "accessory", "id": 31, "name": "handbag" },
    { "supercategory": "accessory", "id": 32, "name": "tie" },
    { "supercategory": "accessory", "id": 33, "name": "suitcase" },
    { "supercategory": "sports", "id": 34, "name": "frisbee" },
    { "supercategory": "sports", "id": 35, "name": "skis" },
    { "supercategory": "sports", "id": 36, "name": "snowboard" },
    { "supercategory": "sports", "id": 37, "name": "sports ball" },
    { "supercategory": "sports", "id": 38, "name": "kite" },
    { "supercategory": "sports", "id": 39, "name": "baseball bat" },
    { "supercategory": "sports", "id": 40, "name": "baseball glove" },
    { "supercategory": "sports", "id": 41, "name": "skateboard" },
    { "supercategory": "sports", "id": 42, "name": "surfboard" },
    { "supercategory": "sports", "id": 43, "name": "tennis racket" },
    { "supercategory": "kitchen", "id": 44, "name": "bottle" },
    { "supercategory": "kitchen", "id": 46, "name": "wine glass" },
    { "supercategory": "kitchen", "id": 47, "name": "cup" },
    { "supercategory": "kitchen", "id": 48, "name": "fork" },
    { "supercategory": "kitchen", "id": 49, "name": "knife" },
    { "supercategory": "kitchen", "id": 50, "name": "spoon" },
    { "supercategory": "kitchen", "id": 51, "name": "bowl" },
    { "supercategory": "food", "id": 52, "name": "banana" },
    { "supercategory": "food", "id": 53, "name": "apple" },
    { "supercategory": "food", "id": 54, "name": "sandwich" },
    { "supercategory": "food", "id": 55, "name": "orange" },
    { "supercategory": "food", "id": 56, "name": "broccoli" },
    { "supercategory": "food", "id": 57, "name": "carrot" },
    { "supercategory": "food", "id": 58, "name": "hot dog" },
    { "supercategory": "food", "id": 59, "name": "pizza" },
    { "supercategory": "food", "id": 60, "name": "donut" },
    { "supercategory": "food", "id": 61, "name": "cake" },
    { "supercategory": "furniture", "id": 62, "name": "chair" },
    { "supercategory": "furniture", "id": 63, "name": "couch" },
    { "supercategory": "furniture", "id": 64, "name": "potted plant" },
    { "supercategory": "furniture", "id": 65, "name": "bed" },
    { "supercategory": "furniture", "id": 67, "name": "dining table" },
    { "supercategory": "furniture", "id": 70, "name": "toilet" },
    { "supercategory": "electronic", "id": 72, "name": "tv" },
    { "supercategory": "electronic", "id": 73, "name": "laptop" },
    { "supercategory": "electronic", "id": 74, "name": "mouse" },
    { "supercategory": "electronic", "id": 75, "name": "remote" },
    { "supercategory": "electronic", "id": 76, "name": "keyboard" },
    { "supercategory": "electronic", "id": 77, "name": "cell phone" },
    { "supercategory": "appliance", "id": 78, "name": "microwave" },
    { "supercategory": "appliance", "id": 79, "name": "oven" },
    { "supercategory": "appliance", "id": 80, "name": "toaster" },
    { "supercategory": "appliance", "id": 81, "name": "sink" },
    { "supercategory": "appliance", "id": 82, "name": "refrigerator" },
    { "supercategory": "indoor", "id": 84, "name": "book" },
    { "supercategory": "indoor", "id": 85, "name": "clock" },
    { "supercategory": "indoor", "id": 86, "name": "vase" },
    { "supercategory": "indoor", "id": 87, "name": "scissors" },
    { "supercategory": "indoor", "id": 88, "name": "teddy bear" },
    { "supercategory": "indoor", "id": 89, "name": "hair drier" },
    { "supercategory": "indoor", "id": 90, "name": "toothbrush" }
  ]
    
    return categories
    
def extract_coco_images_and_json(cocopath,category_id:list,num,output_dir,filename):
    with open(cocopath, 'r') as f:

        json_data = json.load(f)

    # get img_id which includes person(1) category
    image_id_list = []
    for i in range(len(json_data["annotations"])):
        catid = json_data["annotations"][i]["category_id"]
        for id in category_id:
            if catid == id :
                image_id_list.append(json_data["annotations"][i]["image_id"])
            if len(image_id_list) == num :
                break

    # create empty coco form for extraction
    json_data_coco = coco_style_gen()

    coco_annotation = COCO(cocopath)
    cat_ids_list = []
    for i in range(len(image_id_list)):
        # get anno info from image_id extracted above and filter person class
        ann_ids = coco_annotation.getAnnIds(imgIds=[image_id_list[i]], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        
        for j in range(len(anns)):
            annscatid = anns[j]["category_id"]
            for id in category_id:
                if annscatid == id : 
                    json_data_coco["annotations"].append(anns[j])
                    cat_ids_list.append(annscatid)
        
        # get image info from image_id extracted above and append to image field
        img_info = coco_annotation.loadImgs([image_id_list[i]])[0]
        json_data_coco["images"].append(img_info)
        


        # get origin image from image_id extracted above and save to local
        img_file_name = img_info["file_name"]
        img_url = img_info["coco_url"]
        im = Image.open(requests.get(img_url, stream=True).raw)
        cv2.imwrite(output_dir+img_file_name,cv2.cvtColor(np.asarray(im),cv2.COLOR_BGR2RGB))            

    #get category field from each annotation fields and map with original file
    # print(cat_ids_list)
    cat_ids_list = list(set(cat_ids_list))
    cats = coco_annotation.loadCats(coco_annotation.getCatIds())
    cat_ids_list.sort()
    for i in range(len(cat_ids_list)):
        for j in range(len(cats)):
            if cat_ids_list[i] == cats[j]["id"] : json_data_coco["categories"].append({"id":cat_ids_list[i],
                                                                                    "name":cats[j]["name"]})

    with open(output_dir+filename, 'w', encoding='utf-8') as make_file:
        json.dump(json_data_coco, make_file)