import numpy as np
import pickle
from glob import glob
from detectron2.config import get_cfg
import torch
import tensorflow as tf
import os, six
from tensorpack.utils.gpu import get_num_gpu
from pycocotools.coco import COCO
from tensorpack.predict import PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorConverter.config import config, _C
from tensorConverter.convert_d2 import convert_config, convert_weights

from tensorConverter.modeling.generalized_rcnn import ResNetFPNModel

def finalize_configs(class_name:list):
    default_class_name = ["BG"]
    [default_class_name.append(name) for name in class_name]
    """
    Run some sanity checks, and populate some configs from others
    """
    _C.freeze(False)  # populate new keys now
    if isinstance(_C.DATA.VAL, six.string_types):  # support single string (the typical case) as well
        _C.DATA.VAL = (_C.DATA.VAL, )
    if isinstance(_C.DATA.TRAIN, six.string_types):  # support single string
        _C.DATA.TRAIN = (_C.DATA.TRAIN, )

    # finalize dataset definitions ...
    # from dataset import DatasetRegistry
    # datasets = list(_C.DATA.TRAIN) + list(_C.DATA.VAL)
    # _C.DATA.CLASS_NAMES = DatasetRegistry.get_metadata(datasets[0], "class_names")
    # _C.DATA.CLASS_NAMES = ["BG","person","bicycle","car"]
    _C.DATA.CLASS_NAMES = default_class_name
    _C.DATA.NUM_CATEGORY = len(_C.DATA.CLASS_NAMES) - 1
    # _C.DATA.NUM_CATEGORY = 3

    assert _C.BACKBONE.NORM in ['FreezeBN', 'SyncBN', 'GN', 'None'], _C.BACKBONE.NORM
    if _C.BACKBONE.NORM != 'FreezeBN':
        assert not _C.BACKBONE.FREEZE_AFFINE
    assert _C.BACKBONE.FREEZE_AT in [0, 1, 2]

    _C.RPN.NUM_ANCHOR = len(_C.RPN.ANCHOR_SIZES) * len(_C.RPN.ANCHOR_RATIOS)
    assert len(_C.FPN.ANCHOR_STRIDES) == len(_C.RPN.ANCHOR_SIZES)
    # image size into the backbone has to be multiple of this number
    _C.FPN.RESOLUTION_REQUIREMENT = _C.FPN.ANCHOR_STRIDES[3]  # [3] because we build FPN with features r2,r3,r4,r5

    if _C.MODE_FPN:
        size_mult = _C.FPN.RESOLUTION_REQUIREMENT * 1.
        _C.PREPROC.MAX_SIZE = np.ceil(_C.PREPROC.MAX_SIZE / size_mult) * size_mult
        assert _C.FPN.PROPOSAL_MODE in ['Level', 'Joint']
        assert _C.FPN.FRCNN_HEAD_FUNC.endswith('_head')
        assert _C.FPN.MRCNN_HEAD_FUNC.endswith('_head')
        assert _C.FPN.NORM in ['None', 'GN']

        if _C.FPN.CASCADE:
            # the first threshold is the proposal sampling threshold
            assert _C.CASCADE.IOUS[0] == _C.FRCNN.FG_THRESH
            assert len(_C.CASCADE.BBOX_REG_WEIGHTS) == len(_C.CASCADE.IOUS)
    # autotune is too slow for inference
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    ngpu = get_num_gpu()

    if _C.TRAIN.NUM_GPUS is None:
        _C.TRAIN.NUM_GPUS = ngpu
    else:
        if _C.TRAINER == 'horovod':
            assert _C.TRAIN.NUM_GPUS == ngpu
        else:
            assert _C.TRAIN.NUM_GPUS <= ngpu

    _C.freeze()

class D2toTensorflow:
    def __init__(self,cfg,weight_path,dataset_dir,output_pb,export_type = "pb"):
        if type(cfg) == str:
            config = get_cfg()
            config.merge_from_file(cfg)
            self.cfg = config
        else:
            self.cfg = cfg
        self.weight_path = weight_path
        self.dataset_dir = dataset_dir
        self.outdir = "/".join(self.weight_path.split("/")[:-1])+"/"
        # self.output = output_file_name
        self.output_pb = output_pb
        self.cfg_list = []
        self.export_type = export_type
        self.np_converted_dict = {}
        self.D2Convert()
        
    def D2Convert(self):
        tp_cfg = convert_config(self.cfg)
        for k, v in tp_cfg:
            self.cfg_list.append('{}={}'.format(k, v).replace(' ', ''))
        if self.weight_path.split(".")[-1] == "pkl":
            with open(self.weight_path, "rb") as f:
                d2_dict = pickle.load(f)["model"]
            tp_dict = convert_weights(d2_dict, self.cfg)
        elif self.weight_path.split(".")[-1] == "pth":
            d2_dict = torch.load(self.weight_path)["model"]
            layers = [key for key in d2_dict.keys()]
            for i in range(len(layers)):
                d2_dict[layers[i]] = d2_dict[layers[i]].cpu().numpy()
        
            tp_dict = convert_weights(d2_dict, self.cfg)
        self.np_converted_dict = tp_dict    
        self.emitPb()
    
    def emitPb(self):
        config.update_args(self.cfg_list)
        MODEL = ResNetFPNModel() 
        
        if not tf.test.is_gpu_available():
            from tensorflow.python.framework import test_util
            assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
                "Inference requires either GPU support or MKL support!"
        
        assert self.cfg_list, self.dataset_dir
        class_name = []
        annofile = [i for i in glob(self.dataset_dir+"/*.json")][0]
        coco_annotation = COCO(annofile)
        cats = coco_annotation.loadCats(coco_annotation.getCatIds())
        [class_name.append(cat['name']) for cat in cats]
        finalize_configs(class_name)
        
        predcfg = PredictConfig(
            model=MODEL,
            # session_init=SmartInit(self.outdir+self.output),
            session_init=SmartInit(self.np_converted_dict),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])
        
        if self.export_type == "pb":
            ModelExporter(predcfg).export_compact(self.output_pb+".pb", optimize=False)
        elif self.export_type == "serving":
            ModelExporter(predcfg).export_serving(self.output_pb)
        