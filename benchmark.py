import os

import gradio as gr
import numpy as np
import torch

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from tools.engine import Config
from tools.utils.ckpt import load_ckpt
import io


def build_rec_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        # TODO
        elif op_name in ['DecodeImage']:
            op[op_name]['gradio_infer_mode'] = True

        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if cfg['Architecture']['algorithm'] == 'SRN':
                op[op_name]['keep_keys'] = [
                    'image',
                    'encoder_word_pos',
                    'gsrm_word_pos',
                    'gsrm_slf_attn_bias1',
                    'gsrm_slf_attn_bias2',
                ]
            elif cfg['Architecture']['algorithm'] == 'SAR':
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif cfg['Architecture']['algorithm'] == 'RobustScanner':
                op[op_name]['keep_keys'] = [
                    'image', 'valid_ratio', 'word_positons'
                ]
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    return transforms


def get_all_file_names_including_subdirs(dir_path):
    all_file_names = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            all_file_names.append(os.path.join(root, file_name))

    file_names_only = [os.path.basename(file) for file in all_file_names]
    return file_names_only


root_directory = './configs/rec'
yml_Config = get_all_file_names_including_subdirs(root_directory)


def find_file_in_current_dir_and_subdirs(file_name):
    # print(file_name)
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            relative_path = os.path.join(root, file_name)
            return relative_path

def load_model(Model_type="svtrv2_ctc_deepctrl.yml"):
    path = find_file_in_current_dir_and_subdirs(Model_type)
    cfg = Config(path).cfg
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    global_config = cfg['Global']
    char_num = len(getattr(post_process_class, 'character'))
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.eval()
    return model

def predict(input_image, model):
    Model_type = "svtrv2_ctc_deepctrl.yml"
    if os.path.isfile(input_image) and os.path.exists(input_image):
        with open(input_image, 'rb') as f:
            input_image = f.read()

    path = find_file_in_current_dir_and_subdirs(Model_type)

    cfg = Config(path).cfg
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    global_config = cfg['Global']
    char_num = len(getattr(post_process_class, 'character'))
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    transforms = build_rec_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)
    data = {'image': input_image}
    batch = transform(data, ops)
    others = None
    images = np.expand_dims(batch[0], axis=0)
    images = torch.from_numpy(images)
    with torch.no_grad():
        model.eval()
        preds = model(images, others)
    post_result = post_process_class(preds)
    return post_result[0][0], post_result[0][1]


if __name__ == '__main__':
    base_file_path = "/home/deepctrl/liwenju/ocr-train/PaddleOCR/baike_chinese_bgd_1k/cutted_img"
    files = os.listdir(base_file_path)
    total_count = 0
    total_right = 0
    model = load_model()
    for file in files:
        label = file.split('.')[0].split('_')[-1]
        result, confidence = predict(os.path.join(base_file_path, file), model)
        print(result, label, confidence)
        total_count += 1
        if result == label:
            total_right += 1
    print(f"Total count: {total_count}, Total right: {total_right}")
