import docker , os

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
            
def trainserver_start(dataset_path,model_repo,servable_model_repo,labeling_type,project_name,device_id):
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

