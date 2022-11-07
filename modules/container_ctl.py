import docker , os
from glob import glob

def get_container_list(base_url = 'tcp://192.168.0.2:2375'):
    name_list = []
    # client = docker.from_env()
    client = docker.DockerClient(base_url=base_url)
    for i in client.containers.list():
        name = i.name
        name_list.append(name)
    return name_list

def get_port_usage(base_url = 'tcp://192.168.0.2:2375'):
    # client = docker.from_env()
    client = docker.DockerClient(base_url=base_url)
    ports_in_use = {}
    for i in client.containers.list():
        container = client.containers.get(i.id)
        if len(container.ports) == 0: pass
        else:ports_in_use[str(i.name)] = list(container.ports.items())[0][-1][0]["HostPort"]
    return ports_in_use

def stop_container(name,base_url = 'tcp://192.168.0.2:2375'):
    # client = docker.from_env()
    client = docker.DockerClient(base_url=base_url)
    for i in client.containers.list():
        container_name = i.name
        if name == container_name :
            i.stop()

def rm_container(name,base_url = 'tcp://192.168.0.2:2375'):
    client = docker.DockerClient(base_url=base_url)
    for i in client.containers.list():
        container_name = i.name
        if name == container_name :
            i.remove()
            
def train_server_start(dataset_path,
                       model_repo,
                       servable_model_repo,
                       labeling_type,
                       project_name,
                       device_id,
                       base_url = 'tcp://192.168.0.2:2375',
                       serving_host = "192.168.0.3"):
    
    client = docker.DockerClient(base_url=base_url)
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:t-v0.3',
    name = "train_server_"+str(project_name),
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
                    --dataset_dir /workspace/dataset --labeling_type {labeling_type} --project_name {project_name} --serving_host {serving_host}", #--ouput_host {host_model_repo}
    remove = True
    )
    
    return container

def inference_server_start(model_repo_path,port,type = "od",device_id = 0,mode = "explicit",base_url = 'tcp://192.168.0.2:2375'):
    # client = docker.from_env()
    client = docker.DockerClient(base_url=base_url)
    option = ""
    if type == "od":
       model_repo_path =  model_repo_path + "/od"
       ct_name = "inference_server_od"
    else : 
        model_repo_path =  model_repo_path + "/seg"
        ct_name = "inference_server_seg"
    if mode == "poll":
        option = " --repository-poll-secs=10"
        
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:i-v0.1',
    name = ct_name,
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=[str(device_id)], capabilities=[["gpu"]])
    ],
    ports = {'8000/tcp': port},
    shm_size ="12G",
    volumes = {model_repo_path:{'bind':"/models",'mode':"rw"}},
    command = f"tritonserver --model-repository=/models --model-control-mode={mode}"+option,
    remove = True
    )
    return container

