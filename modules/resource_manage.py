import subprocess , os
import json
from modules.container_ctl import get_container_list
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from glob import glob

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_current_learning_status(project_name,model_repository):
    result = {"project_name":"","learn_iteration":"","status":"","eta_sec":""}
    prjs = [i for i in glob(model_repository+"/*") if project_name in i.split("/")[-1]]
    if "_" in prjs[-1].split("/")[-1]:
        latest_iter = str(prjs[-1].split("/")[-1].split("_")[-1])
    else : 
        latest_iter = "0"
    for i in range(len(prjs)):
        if ("model_final.pth" not in [i.split("/")[-1] for i in glob(prjs[i] + "/*")]) and project_name not in [i.split("_")[-1] for i in get_container_list()]:
            result["project_name"] = project_name
            result["learn_iteration"] = latest_iter
            result["status"] = "Failed"
            result["eta_sec"] = "0"
        elif ("model_final.pth" in [i.split("/")[-1] for i in glob(prjs[i] + "/*")]) and project_name not in [i.split("_")[-1] for i in get_container_list()]:
            result["project_name"] = project_name
            result["learn_iteration"] = latest_iter
            result["status"] = "Success"
            result["eta_sec"] = "0"
        elif ("model_final.pth" not in [i.split("/")[-1] for i in glob(prjs[i] + "/*")]) and project_name in [i.split("_")[-1] for i in get_container_list()]:
            result["project_name"] = project_name
            result["learn_iteration"] = latest_iter
            result["status"] = "Running"
            metric_path = os.path.join(prjs[i] + "/","metrics.json")
            f = open(metric_path, 'r')
            lines = f.readlines()
            json_data = []
            for line in lines:
                json_ins = json.loads(line)
                json_data.append(json_ins)
            result["eta_sec"] = str(int(json_data[-1]["eta_seconds"]))
    return result

def get_model_status(learning_status_result,servable_model_repo):
    result={"model_name":"","state":""}
    if learning_status_result["status"] == "Success":
        if learning_status_result["learn_iteration"] == str(0):
            if learning_status_result["project_name"] in [i.split("/")[-1] for i in glob(servable_model_repo+"/*")]:
                result["model_name"] = learning_status_result["project_name"]
                result["state"] = "READY" 
            else:
                if learning_status_result["project_name"] not in get_container_list():
                    result["model_name"] = learning_status_result["project_name"]
                    result["state"] = "Failed"
                else:    
                    result["model_name"] = learning_status_result["project_name"]
                    result["state"] = "PROCESSING"
        else:
            if learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"] == any([i.split("/")[-1] for i in glob(servable_model_repo+"/*")]):
                result["model_name"] = learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"]
                result["state"] = "READY"
            else :
                if learning_status_result["project_name"] not in get_container_list():
                    result["model_name"] = learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"]
                    result["state"] = "Failed"
                else:
                    result["model_name"] = learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"]
                    result["state"] = "PROCESSING"
        
    elif learning_status_result["status"] == "Running":
        if learning_status_result["learn_iteration"] == str(0):
            result["model_name"] = learning_status_result["project_name"]
        else:
            result["model_name"] = learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"]
        result["state"] = "PROCESSING"
    elif learning_status_result["status"] == "Failed":
        if learning_status_result["learn_iteration"] == str(0):
            result["model_name"] = learning_status_result["project_name"]
        else:
            result["model_name"] = learning_status_result["project_name"]+"_"+learning_status_result["learn_iteration"]
        result["state"] = "Failed"
        
    return result
            
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
            
def get_gpu_proc(nvidia_smi_path='nvidia-smi'):
    result_list = []
    
    cmd = '%s' % (nvidia_smi_path)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    for i in range(len(lines)):
        if "GPU   GI   CI" in lines[i]:
            proc_info_index = i
    proc_lines = lines[proc_info_index:-1]
    if "No running processes found" in proc_lines[-1]: 
        return result
    else:
        processes = [proc_lines[3:][i].replace(" ","_").strip("|") for i in range(len(proc_lines[3:]))]
        process_refine = [list(filter(None,process.split("_"))) for process in processes]
        for i in range(len(process_refine)):
            result = {"GPU_ID":'',"Process_name":'',"GPU_Memory_Usage":''}
            for j in range(len(process_refine[i])):          
                if j == 0 : result["GPU_ID"] = process_refine[i][j]
                elif j == 5 : result["Process_name"] = process_refine[i][j]
                elif j == 6 : result["GPU_Memory_Usage"] = process_refine[i][j]
            result_list.append(result)
        return result_list

def get_model_info(model_name = "all",url = "localhost",port = 8000):
    triton_client = httpclient.InferenceServerClient(url= url+":"+str(port), verbose=False)
    models_state = triton_client.get_model_repository_index()
    result = []
    if model_name == "all":
        return models_state
    else:    
        for status in models_state:
            if status["name"] == model_name :
                return status
        
def is_gpu_trainable(device_id = 1,fraction = 0.5):
    gpu_mem_usage_total = 0
    is_trainable = False
    for proc in get_gpu_proc():
        if proc["GPU_ID"] == str(device_id):
            gpu_mem_usage_total += int(proc["GPU_Memory_Usage"].strip("MiB"))
    total_mem = [info['memory.total'] for info in get_gpu_info() if info['index'] == str(device_id)][0]
    if int(total_mem*fraction) <= (total_mem - gpu_mem_usage_total) : is_trainable = True
    else: is_trainable = False
    return is_trainable

def get_free_gpu_mem(device_id = 1):
    gpu_mem_usage_total = 0
    is_trainable = False
    for proc in get_gpu_proc():
        if proc["GPU_ID"] == str(device_id):
            gpu_mem_usage_total += int(proc["GPU_Memory_Usage"].strip("MiB"))
    total_mem = [info['memory.total'] for info in get_gpu_info() if info['index'] == str(device_id)][0]
    return (total_mem - gpu_mem_usage_total) 

def model_ctl(ctl,model_name,host= "localhost",port = 8000):
    triton_client = httpclient.InferenceServerClient(url=host+":"+str(port), verbose=False)
    if ctl == "load":
        triton_client.load_model(model_name)
    if ctl == "unload":
        triton_client.unload_model(model_name)
    
    return [stats for stats in triton_client.get_model_repository_index() if stats["name"] == model_name]