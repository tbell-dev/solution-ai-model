import json
import os
from glob import glob

def get_loss(metric_path):
    metric_path = os.path.join(metric_path,"metrics.json")
    f = open(metric_path, 'r')
    lines = f.readlines()
    json_data = []
    for line in lines:
        json_ins = json.loads(line)
        json_data.append(json_ins)

    return json_data[-1]

def model_validation(model_repo,current_prj_name):
    prjs = [i for i in glob(model_repo+"/*")]
    print(prjs)
    current_loss,current_prj_indx = 0,None
    previous_losses = []
    for i in range(len(prjs)):
        if current_prj_name == prjs[i].split("/")[-1]:
            current_loss = get_loss(prjs[i])["total_loss"]
            current_prj_indx = i
        else :
            previous_losses.append(get_loss(prjs[i])["total_loss"])
            
    if min(previous_losses) < current_loss : 
        return "model fail"
    elif min(previous_losses) >= current_loss : 
        return prjs[current_prj_indx]+"/model_final.pth"