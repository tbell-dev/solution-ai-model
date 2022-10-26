import docker

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
            
def trainserver_start(dataset_path,host_model_repo,labeling_type,project_name,device_id):
    client = docker.from_env()
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:t-v0.1',
    name = "trainserver_"+str(project_name),
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=[str(device_id)], capabilities=[["gpu"]])
    ],
    volumes = {'/home/tbelldev/workspace/autoLabeling/api_test/tool':{'bind':"/workspace/src",'mode':"rw"},
               dataset_path:{"bind":"/workspace/dataset","mode":"rw"},
               "/home/tbelldev/workspace/autoLabeling/api_test/model_repo":{"bind":"/workspace/output","mode":"rw"}
               },
    command = f"conda run --no-capture-output -n detectron2 \
                python src/train_container.py \
                    --dataset_dir /workspace/dataset --labeling_type {labeling_type} --project_name {project_name} --ouput_host {host_model_repo}",
    remove = True
    )
    
    return container

def inference_server_start(model_repo_path,port,project_name,device_id):
    client = docker.from_env()
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:i-v0.1',
    name = "inference_server_"+str(project_name),
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=[str(device_id)], capabilities=[["gpu"]])
    ],
    ports = {'8000/tcp': port},
    shm_size ="1G",
    volumes = {model_repo_path:{'bind':"/models",'mode':"rw"}},
    command = "tritonserver --model-repository=/models",
    remove = True
    )
    
    return container

