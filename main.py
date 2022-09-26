from fastapi import FastAPI
from typing import Optional
from glob import glob


# docker 관련 라이브러리 필요 (for model serving, active learning)
from inference import *

DATA_ROOT_DIR = "project_images/"

app = FastAPI()

def get_user_uploaded_images(task_id,project_id):
    pattern = "*.{jpg,jpeg,png,bmp}"
    iamge_dir = DATA_ROOT_DIR + project_id+"/"+"changed/"
    image_lists = [i for i in glob(iamge_dir+pattern)]
    
    return image_lists 

@app.get("/")
async def root():
  return {"message": "this is autolabeing api test"}


#task_id = image 개별 id

#Inference
@app.get("/autolabeling/{task_id}/{project_id}/{labeling_type}")
def get_auto_labeling_result(task_id:str ,project_id:str,labeling_type:Optional[str] = "bbox"):
    
    # if labeling_type == "bbox":
    #     pass
    # elif labeling_type == "polygon":
    #     pass
    # elif labeling_type == "segment":
    #     pass
    #get {image,port,class name} using task_id, project_id
    image_list = get_user_uploaded_images(task_id,project_id)
    # anno = inference_local()
    for image in image_list :  response =  inference_triton(image,"person",labeling_type)
    
    return response


#active learning

#model export
