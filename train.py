# Time: 2022/4/26  19:14
import torch
from model import DETRdemo
detr = DETRdemo(num_classes=91)
# 这里是直接用别人训练好的模型

###########################################################
#            下载好之后把参数放到weights里面并重命名           #
###########################################################
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();

