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
    prjs = [i for i in glob(model_repo+"/*") if current_prj_name in i.split("/")[-1]]
    if len(prjs) == 1 : 
        return prjs[0]+"/model_final.pth"
    elif len(prjs) == 0 :
        return model_repo+"/"+current_prj_name+"/model_final.pth"
    else :
        latest_result = prjs[-1]
        previous_result = prjs[:-1]
        latest_loss = get_loss(latest_result)["total_loss"]
        
        previous_losses = []
        for i in range(len(previous_result)):
            previous_losses.append(get_loss(previous_result[i])["total_loss"])
                
        if min(previous_losses) < latest_loss : 
            min_loss = min(previous_losses)
            idx = previous_losses.index(min_loss)
            return previous_result[idx]+"/model_final.pth"
        
        elif min(previous_losses) >= latest_loss : 
            return latest_result+"/model_final.pth"