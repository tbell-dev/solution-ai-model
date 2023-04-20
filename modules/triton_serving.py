import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import time
import tritonclient.http as httpclient
from PIL import Image
import numpy as np
from modules.labels import COCO_NAMES
import cv2

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def inference(model_name,image_file,task_type,port=8000,host = "localhost",print_output=True):
    e = time.time()
    pred = client_v2(image_file, model_name,task_type,port,host,print_output)
    s = time.time()
    print('speed:', (s - e))
    return pred

# def client_v1(image_file, model_name,task_type,port=8000,host = "localhost",print_output=True):
#     img = np.array(Image.open(image_file))
#     img = np.ascontiguousarray(img.transpose(2, 0, 1))
#     # Define model's inputs
#     inputs = []
#     inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
#     inputs[0].set_data_from_numpy(img)
#     # Define model's outputs
#     outputs = []
#     if task_type == "seg":
#         outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
#         outputs.append(httpclient.InferRequestedOutput('classes__1'))
#         outputs.append(httpclient.InferRequestedOutput('masks__2'))
#         outputs.append(httpclient.InferRequestedOutput('scores__3'))
#         outputs.append(httpclient.InferRequestedOutput('shape__4'))
#     elif task_type == "od":
#         outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
#         outputs.append(httpclient.InferRequestedOutput('classes__1'))
#         outputs.append(httpclient.InferRequestedOutput('scores__2'))
#         outputs.append(httpclient.InferRequestedOutput('shape__3'))
        
#     # Send request to Triton server
#     triton_client = httpclient.InferenceServerClient(
#         url=host+":"+str(port), verbose=False)
#     results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
#     response_info = results.get_response()
#     outputs = {}
    
#     for output_info in response_info['outputs']:
#         output_name = output_info['name']
#         outputs[output_name] = results.as_numpy(output_name)
        
#     return outputs


def client_v2(image_file, model_name,task_type,port=8000, host = "localhost",print_output=False):
    if task_type == "seg":
        if type(model_name) == str:
            pass 
        else:
            model_name = "infer_pipeline_"+str(model_name)
        if type(image_file) == bytes:
            image_bytes = image_file
        else:
            with open(image_file, 'rb') as fi:
                image_bytes = fi.read()
        image_bytes = np.array([image_bytes], dtype=np.bytes_)
        # Define model's inputs
        inputs = []
        inputs.append(httpclient.InferInput('IMAGE_BYTES', image_bytes.shape, "BYTES"))
        inputs[0].set_data_from_numpy(image_bytes)
        # Define model's outputs
        outputs = []

        outputs.append(httpclient.InferRequestedOutput('BBOXES'))
        outputs.append(httpclient.InferRequestedOutput('CLASSES'))
        outputs.append(httpclient.InferRequestedOutput('MASKS'))
        outputs.append(httpclient.InferRequestedOutput('SCORES'))
    elif task_type == "od":
        img = np.array(Image.open(image_file))
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # Define model's inputs
        inputs = []
        inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
        inputs[0].set_data_from_numpy(img)
        # Define model's outputs
        outputs = []

        outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
        outputs.append(httpclient.InferRequestedOutput('classes__1'))
        outputs.append(httpclient.InferRequestedOutput('scores__2'))
        outputs.append(httpclient.InferRequestedOutput('shape__3'))
    elif task_type == "keypoint":
        img = np.array(Image.open(image_file))
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # Define model's inputs
        inputs = []
        inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
        inputs[0].set_data_from_numpy(img)
        # Define model's outputs
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
        outputs.append(httpclient.InferRequestedOutput('classes__1'))
        outputs.append(httpclient.InferRequestedOutput('keypoint_heatmaps__2'))
        outputs.append(httpclient.InferRequestedOutput('keypoints__3'))
        outputs.append(httpclient.InferRequestedOutput('scores__4'))
        outputs.append(httpclient.InferRequestedOutput('shape__5'))

    # Send request to Triton server
    triton_client = httpclient.InferenceServerClient(
        url=host+":"+str(port), verbose=False)
    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)

    return outputs
    
def get_key_from_value(dict,value):
   return [k for k, v in dict.items() if v in value]

def infer_result_filter(pred,task_type,score:float,class_name:list):
    # get filtered index
    idx_list, new_pred = [], {}
    
    for key in list(pred.keys()):
        new_pred[key] = []
        
    if task_type == "seg":
        for i in range(len(pred[list(pred.keys())[0]])):
            if pred[list(pred.keys())[0]][i] >= score and pred[list(pred.keys())[2]][i] in get_key_from_value(COCO_NAMES,class_name):
                idx_list.append(i)
            
        # filter result
        for idx in idx_list: 
            for key in list(pred.keys())[:4]:
                if len(new_pred[key]) == 0:
                    new_pred[key] = [pred[key][idx]]
                else : new_pred[key].append(pred[key][idx])
                
    elif task_type == "od":
        for i in range(len(pred[list(pred.keys())[2]])):
            if pred[list(pred.keys())[2]][i] >= score and pred[list(pred.keys())[1]][i] in get_key_from_value(COCO_NAMES,class_name):
                idx_list.append(i)
            
        # filter result
        for idx in idx_list: 
            for key in list(pred.keys())[:3]:
                if len(new_pred[key]) == 0:
                    new_pred[key] = [pred[key][idx]]
                else : new_pred[key].append(pred[key][idx])
            
        
        new_pred[list(pred.keys())[-1]] = pred[list(pred.keys())[-1]]
    
    return new_pred

def infer_result_filter_conf(pred,task_type,score:float):
    idx_list, new_pred = [], {}
    
    for key in list(pred.keys()):
        new_pred[key] = []
        
    if task_type == "seg":
        for i in range(len(pred[list(pred.keys())[0]])):
            if pred[list(pred.keys())[0]][i] >= score:
                idx_list.append(i)
            
        # filter result
        for idx in idx_list: 
            for key in list(pred.keys())[:4]:
                if len(new_pred[key]) == 0:
                    new_pred[key] = [pred[key][idx]]
                else : new_pred[key].append(pred[key][idx])
                
    elif task_type == "od":
        for i in range(len(pred[list(pred.keys())[2]])):
            if pred[list(pred.keys())[2]][i] >= score:
                idx_list.append(i)
            
        # filter result
        for idx in idx_list: 
            for key in list(pred.keys())[:3]:
                if len(new_pred[key]) == 0:
                    new_pred[key] = [pred[key][idx]]
                else : new_pred[key].append(pred[key][idx])

    elif task_type == "keypoint":
        for i in range(len(pred[list(pred.keys())[4]])):
            if pred[list(pred.keys())[4]][i] >= score:
                idx_list.append(i)
            
        # filter result
        for idx in idx_list: 
            for key in list(pred.keys())[:5]:
                if len(new_pred[key]) == 0:
                    new_pred[key] = [pred[key][idx]]
                else : new_pred[key].append(pred[key][idx])
            
        
        new_pred[list(pred.keys())[-1]] = pred[list(pred.keys())[-1]]
    
    return new_pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help="image path from local filesystem", required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', type=str ,default="od", required=True)
    parser.add_argument('--mode', default="single", choices=["single",'sequential', 'concurrent'])
    parser.add_argument('--num-reqs', default='1')
    parser.add_argument('--conf', type=float , default=0.5)
    parser.add_argument('--class_name', type=str ,default="person")
    parser.add_argument('--print-output', type=bool ,default=True)
    parser.add_argument('--serving-port', type=int ,default=8000)
    parser.add_argument('--serving-host', type=str ,default="localhost")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    image_path = args.image
    model_name = args.model
    task_type = str(args.task)
    mode = args.mode
    confidence = args.conf
    class_name =  args.class_name
    # n_reqs = int(args.num_reqs)
    port = int(args.serving_port)
    host = int(args.serving_host)
    # model_name = "mask_rcnn"
    # image_path = "/home/tbelldev/workspace/autoLabeling/dataset/coco_person/images/val/000000408774.jpg"

    pred = inference(model_name,image_path,task_type,port=port,host=host)
    result = infer_result_filter(pred,task_type,confidence,class_name)
    
    if args.print_output : print(result)