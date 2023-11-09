import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from config.dataset_class import activity_dict
# from gsm_lib import opts
import yaml 


with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

output_path = config['dataset']['testing']['output_path']
nms_thresh = config['testing']['nms_thresh']

from evaluation.eval_detection import ANETdetection
anet_detection = ANETdetection(
    ground_truth_filename="./evaluation/thumos_new.json",
    prediction_filename="./data/thumos2014/test_output//detection_result_nms0.6.json",
    subset='validation', verbose=True, check_status=False)

anet_detection.evaluate()

mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
print(results)
with open(os.path.join(output_path, 'results.txt'), 'a') as fobj:
    fobj.write(f'{results}\n')