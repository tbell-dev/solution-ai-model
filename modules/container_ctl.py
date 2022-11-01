import docker , os
import subprocess
import json
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

def get_container_list():
    name_list = []
    client = docker.from_env()
    for i in client.containers.list():
        name = i.name
        name_list.append(name)
    return name_list

def get_port_usage():
    client = docker.from_env()
    ports_in_use = {}
    for i in client.containers.list():
        container = client.containers.get(i.id)
        if len(container.ports) == 0: pass
        else:ports_in_use[str(i.name)] = list(container.ports.items())[0][-1][0]["HostPort"]
    return ports_in_use

def stop_container(name):
    client = docker.from_env()
    for i in client.containers.list():
        container_name = i.name
        if name == container_name :
            i.stop()

def rm_container(name):
    client = docker.from_env()
    for i in client.containers.list():
        container_name = i.name
        if name == container_name :
            i.remove()
            
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

def get_model_info(model_name,url = "localhost",port = 8000):
    triton_client = httpclient.InferenceServerClient(url= url+":"+str(port), verbose=False)
    models_state = triton_client.get_model_repository_index()
    for status in models_state:
        if status["name"] == model_name :
            return status
            
def train_server_start(dataset_path,model_repo,servable_model_repo,labeling_type,project_name,device_id):
    client = docker.from_env()
    pwd = os.getcwd()
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:t-v0.2',
    name = "trainserver_"+str(project_name),
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=[str(device_id)], capabilities=[["gpu"]])
    ],
    volumes = {os.getcwd()+'/tool':{'bind':"/workspace/src",'mode':"rw"},
               dataset_path:{"bind":"/workspace/dataset","mode":"rw"},
               servable_model_repo:{"bind":"/workspace/models","mode":"rw"}, 
               model_repo:{"bind":"/workspace/output","mode":"rw"} 
               },
    command = f"conda run --no-capture-output -n detectron2 \
                python src/container_pipeline.py \
                    --dataset_dir /workspace/dataset --labeling_type {labeling_type} --project_name {project_name}", #--ouput_host {host_model_repo}
    remove = True
    )
    
    return container

# 10초 마다 한번 씩 servable model repository에 변경사항을 모니터링하여 모델 load / unload 수행 --> 특별한 메뉴얼 조치없이 초기 실행 이후 계속 실행
def inference_server_start(model_repo_path,port,container_cnt,device_id):
    client = docker.from_env()
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:i-v0.1',
    name = "inference_server_"+str(container_cnt),
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=[str(device_id)], capabilities=[["gpu"]])
    ],
    ports = {'8000/tcp': port},
    shm_size ="30G",
    volumes = {model_repo_path:{'bind':"/models",'mode':"rw"}},
    command = "tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=10",
    remove = True
    )
    
    return container

