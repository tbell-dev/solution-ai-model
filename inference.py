#custom helper module
from modules.post_proccessing import *

# make data format for REST api
from modules.formatter import *

# triton serving client
from modules.triton_serving import *
    
def inference_triton(image_path,model_name:str,confidence,port=8000,label_type = "bbox"):
    # inference using triton serving container , with httpclient 
    if label_type == "bbox":
        task_type = "od"
        # model_name = "faster_rcnn"
        pred = inference(model_name,image_path,task_type,port=8000)
        result = infer_result_filter_conf(pred,task_type,confidence)
        
    elif label_type == "polygon" or label_type == "segment":
        task_type = "seg"
        # model_name = "infer_pipeline"
        pred = inference(model_name,image_path,task_type,port=8001)
        result = infer_result_filter_conf(pred,task_type,confidence)
    
    response_data = coco_format_inverter(result)
    # return result
    return response_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help="image path from local filesystem", required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', type=str ,default="od", required=True)
    parser.add_argument('--conf', type=float , default=0.6)
    parser.add_argument('--class_name', type=str ,default="person")
    parser.add_argument('--serving-port', type=int ,default=8000)
    parser.add_argument('--serving-host', type=str ,default='localhost')
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
