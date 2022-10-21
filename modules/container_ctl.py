import docker

client = docker.from_env()

def trainserver_start(dataset_path,labeling_type,project_name):
    container = client.containers.run(
    image = 'tbelldev/sslo-ai:t-v0.1',
    name = "trainserver",
    detach=True,
    runtime="nvidia",
    device_requests=[
        docker.types.DeviceRequest(device_ids=["0","1"], capabilities=[["gpu"]])
    ],
    volumes = {'/home/tbelldev/workspace/autoLabeling/api_test/tool':{'bind':"/workspace/src",'mode':"rw"},
               dataset_path:{"bind":"/workspace/dataset","mode":"rw"},
               "/home/tbelldev/workspace/autoLabeling/api_test/model_repo":{"bind":"/workspace/output","mode":"rw"}
               },
    command = f"conda run --no-capture-output -n detectron2 \
                python src/train_container.py \
                    --dataset_dir /workspace/dataset --labeling_type {labeling_type} --project_name {project_name}",
    remove = True
    )
    
    return container