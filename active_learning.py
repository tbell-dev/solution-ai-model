from tool.augmentation import augmentator
# from tool.train import trainer
import argparse
# import docker
from modules.container_ctl import train_server_start, get_container_list
import json, glob,os

def define_label_type(jsonpath):
    jsonfile = [i for i in glob.glob(jsonpath+"/*.json")][0]
    with open(jsonfile, 'r') as f: user_labeled_data = json.load(f)
    if len(user_labeled_data["annotations"][0]["segmentation"]) > 0 : return "polygon"
    elif len(user_labeled_data["annotations"][0]["segmentation"]) == 0 : return "bbox"

class activeLearning:
    """
    self.dataset_path # 데이터 증강 이전에 사용자들이 라벨링 한 원본 데이터 셋 경로 
    self.aug_dataset_path  # 학습에 사용될 실제 데이터 셋의 경로
    self.project_name # 프로젝트별 고유 식별 코드
    self.labeling_type # bbox or polygon 
    self.v # 데이터 셋 train, val 로 나누는 비율 ex. 0.7 -> train 70% , val 30%
    self.iter  # 데이터 증강 개수 설정
    self.model_repository # 학습 컨테이너에 마운팅 되어 학습만 완료 된 상태의 모델 폴더의 경로 
    self.servable_model_path # 학습 컨테이너에 마운팅 되어 컨테이너에서 모든 프로세스가 끝난 후 
                               inference 될수있는 형태로 변환 된 모델 저장소의 경로
    self.device_id # 학습 시 지정할 GPU 디바이스 아이디 (0,1)
    self.base_url  # 컨테이너를 띄울 호스트 서버 주소:도커 데몬 포트
    self.serving_host # 학습이 완료 된 후 torchscript로 변환된 모델을 triton 서빙 구조로 만들고, 서빙하기 위한 호스트 
    """
    def __init__(self,
                 dataset_dir,
                 project_name,
                #  labeling_type,
                 servable_model_path,
                 model_repository,
                 split = 0.7,
                 aug_iter = 10,
                 model_name = "",
                 epoch = 1000,
                 lr = 0.00025,
                 batch_size = 2,
                 device_id = 1,
                 base_url = 'tcp://192.168.0.2:2375',
                 serving_host = "172.17.0.1",
                 labeling_type = None):
                
        self.dataset_path = str(dataset_dir)+"/" 
        self.aug_dataset_path = str(self.dataset_path) + "augmented/" 
        self.project_name = str(project_name) 
        self.serving_host = serving_host
        
        # get labeling_type from labeled data
        if labeling_type == None:
            self.labeling_type = define_label_type(self.dataset_path) 
        else:
            self.labeling_type = labeling_type
            
        if self.labeling_type == "bbox": self.task_type = "od"
        elif self.labeling_type == "polygon": self.task_type = "seg"
        
        self.v = float(str(split))
        self.iter = int(aug_iter)
        self.model_repository = str(model_repository)
        self.servable_model_path = str(servable_model_path)
        self.device_id = int(device_id)
        self.base_url = base_url

        self.model_name = model_name
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.activate()
        
    def activate_aug(self):
        augmentator(self.dataset_path,
                    self.aug_dataset_path,
                    labeling_type = self.task_type,
                    iter = self.iter)
        
    def activate_train_validation_export(self):
        if "." in self.aug_dataset_path:
            self.aug_dataset_path = "/".join(os.getcwd().split("/")[:-1])+self.aug_dataset_path.strip("..")
            self.model_repository = "/".join(os.getcwd().split("/")[:-1])+"/"+self.model_repository
            self.servable_model_path = "/".join(os.getcwd().split("/")[:-1])+"/"+self.servable_model_path
            
        train_server_start(self.aug_dataset_path,
                          self.model_repository, 
                          self.servable_model_path,
                          self.labeling_type,
                          self.model_name,
                          self.epoch,
                          self.lr,
                          self.batch_size,
                          self.project_name,
                          self.device_id,
                          serving_host = self.serving_host,
                          base_url = self.base_url)
    def activate(self):
        self.activate_aug()
        self.activate_train_validation_export()
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',help="image & json path from local filesystem, root of train/ & val/", required=True)
    parser.add_argument('--project_name', type=str ,default="0", required=True)
    parser.add_argument('--labeling_type', type=str ,default="bbox", required=True)
    parser.add_argument('--split',  type=float ,default=0.7)
    parser.add_argument('--mode',  type=str ,default="container")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    dataset_dir = args.dataset_dir
    labeling_type = args.labeling_type
    project_name = args.project_name
    v = args.split
    