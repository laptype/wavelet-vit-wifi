#!/usr/bin/env python3

import os
import torch

import sys
sys.path.append('/home/wang_f/code/wifi_violence_code')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from scripts.utils import *

# tmux new -s wifi
# tmux a -t wifi
# /home/WangFei/anaconda3/envs/torch/bin/python3 -u /home/WangFei/code/wifi-violence-code/scripts/vit_aug_run.py

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 1

    # os.system('tmux a -t wifi_vio')

    config = DatasetDefaultConfig()

    model_list = [

        # ('wavevit_waveres_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        #
        # ('wavevit_waveres_4_b_32_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_8_b_32_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_0_b_32_0.4_0.1', 'vit_span_cls_raw', 64),

        ('wavevit_waveres_8_s_16_0.4_0.1_0.9', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_8_s_16_0.4_0.1_0.7', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_8_s_16_0.4_0.1_0.5', 'vit_span_cls_raw', 64),
        # ('wavevit_waveres_8_s_16_0.4_0.1_0.3', 'vit_span_cls_raw', 64),
        ('wavevit_waveres_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        ('wavevit_waveres_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
    ]

    config.dataset_list.append(f'WiVioAUG_i-window-w-s')


    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:

            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]

            log_name = 'day_2_27'
            tab = 'day_2_27'
            datasource_path = '/home/wang_f/data/wifi_violence_dataset/wifi_violence_processed_loc/'

            log_path = os.path.join('/home/wang_f/code/wifi_violence_code/log/', log_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            os.system(
                'bash /home/wang_f/code/wifi_violence_code/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            )

