from modules.export_to_torchscript import model_export_to_ts
import yaml,os,shutil

DEFAULT_YAML_PATH = "/home/tbelldev/workspace/autoLabeling/api_test/modules/yaml/"
DEFAULT_MODEL_PATH = "/home/tbelldev/workspace/autoLabeling/api_test/models/"
def make_yaml(task_type,weight_path):
    file_name = ""
    new_cfg = {"_BASE_":"","MODEL":{"WEIGHTS":""}}
    if task_type == "seg":
        file_name = "custom_mask_rcnn_R_101_FPN_3x"
        with open(DEFAULT_YAML_PATH+file_name+'.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif task_type == "od":
        file_name = "custom_faster_rcnn_R_101_FPN_3x"
        with open(DEFAULT_YAML_PATH+file_name+'.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
    new_cfg["MODEL"]["WEIGHTS"] = weight_path
    new_cfg["_BASE_"] =  config["_BASE_"]
    
    prj_name = weight_path.split("/")[-2]
    
    with open(DEFAULT_YAML_PATH+file_name+"_"+prj_name+".yaml", 'w') as fw:
        yaml.dump(new_cfg, fw)
    
    return DEFAULT_YAML_PATH+file_name+"_"+prj_name+".yaml"

def configure_model_dir(task_type,cfg_file,sample_image,output_dir_name):
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
            with open(DEFAULT_MODEL_PATH+"infer_pipeline_"+output_dir_name+'/config.pbtxt') as f:
                txt = f.read()
        if 'mask_rcnn' in txt :
            new_txt = txt.replace('mask_rcnn',output_dir_name)
            with open(DEFAULT_MODEL_PATH+"infer_pipeline_"+output_dir_name+'/config.pbtxt',"w") as f:
                f.write(new_txt)
                
    elif task_type == "od":
        shutil.copy(DEFAULT_MODEL_PATH+"faster_rcnn/config.pbtxt",DEFAULT_MODEL_PATH+output_dir_name+"/config.pbtxt" )   

    model_export_to_ts(cfg_file,sample_image,deploy_dir)