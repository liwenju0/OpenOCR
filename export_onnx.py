import os

import gradio as gr
import numpy as np
import torch
import onnxruntime as ort

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


class ExportConfig:
    def __init__(self):
        self.root_directory = './configs/rec'
        self.model_type = "svtrv2_ctc_deepctrl_distill_resnet.yml"
        self.base_file_path = "/home/deepctrl/liwenju/ocr-train/PaddleOCR/baike_chinese_bgd_1k/cutted_img"
        self.onnx_model_path = "/home/deepctrl/liwenju/ocr-train/OpenOCR/output/svtrv2_ctc_deepctrl_distill/model.onnx"
        self.test_data_path = "/home/deepctrl/liwenju/ocr-train/OpenOCR/test_data"
        
    @property
    def sample_image_path(self):
        return f"{self.test_data_path}/1.jpg"


def find_file_in_current_dir_and_subdirs(file_name):
    # print(file_name)
    for root, dirs, files in os.walk('.'):
        if file_name in files:
            relative_path = os.path.join(root, file_name)
            return relative_path

def load_model(config=None):
    if config is None:
        config = ExportConfig()
    path = find_file_in_current_dir_and_subdirs(config.model_type)
    cfg = Config(path).cfg
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    global_config = cfg['Global']
    char_num = len(getattr(post_process_class, 'character'))
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.eval()
    return model

def predict(input_image, model, config=None):
    if config is None:
        config = ExportConfig()
    input_image = config.sample_image_path
    if os.path.isfile(input_image) and os.path.exists(input_image):
        with open(input_image, 'rb') as f:
            input_image = f.read()

    path = find_file_in_current_dir_and_subdirs(config.model_type)

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
    # print(images.shape)
    with torch.no_grad():
        model.eval()
        preds = model(images, others)
        post_result = post_process_class(preds)
        print(f"pth模型：预测结果: {post_result}")
        torch.onnx.export(
            model, 
            images, 
            config.onnx_model_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 3: 'width'}, 'output': {0: 'batch_size'}},
            opset_version=13)
        print("Exported model.onnx")
    post_result = post_process_class(preds)
    return post_result[0][0], post_result[0][1]

def onnx_predict(image_path, onnx_path, config=None):
    if config is None:
        config = ExportConfig()
    # 加载配置
    cfg = Config(find_file_in_current_dir_and_subdirs(config.model_type)).cfg
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    global_config = cfg['Global']
    
    # 预处理
    transforms = build_rec_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)
    
    with open(image_path, 'rb') as f:
        # print(f"读取图片: {image_path}")
        input_image = f.read()
    
    data = {'image': input_image}
    batch = transform(data, ops)
    images = np.expand_dims(batch[0], axis=0)
    # # 将images存为pickle
    # import pickle
    
    # # 确保输出目录存在
    # import os
    # output_dir = "/home/deepctrl/liwenju/ocr-train/OpenOCR/test_data/"
    # # os.makedirs(output_dir, exist_ok=True)
    
    # # 将images保存为pickle文件
    # pickle_path = os.path.join(output_dir, "images.pkl")
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(images, f)
    
    # print(f"已将images保存为pickle文件: {pickle_path}")
    
    # 加载ONNX模型并进行推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: images}
    ort_outs = ort_session.run(None, ort_inputs)
    # print(f"ort_outs: {ort_outs}, type: {type(ort_outs)}, len: {len(ort_outs)}, shape: {ort_outs[0].shape}")
    # 获取ort_outs[0]这个shape: (1, 64, 8379)的64个最大的id
    top_ids = np.argmax(ort_outs[0], axis=-1)
    # print(f"top_ids: {top_ids}, shape: {top_ids.shape}")
    # 后处理
    post_result = post_process_class(ort_outs[0])
    result, confidence = post_result[0][0], post_result[0][1]
    
    print(f"onnx模型：预测结果: {result}")
    print(f"onnx模型：置信度: {confidence}")
    
    return result, confidence


def export(config=None):
    if config is None:
        config = ExportConfig()
    files = os.listdir(config.base_file_path)
    total_count = 0
    model = load_model(config)
    for file in files:
        label = file.split('.')[0].split('_')[-1]
        result, confidence = predict(os.path.join(config.base_file_path, file), model, config)
        return

def onnx_test(config=None):
    if config is None:
        config = ExportConfig()
    onnx_predict(config.sample_image_path, config.onnx_model_path, config)

if __name__ == '__main__':
    config = ExportConfig()
    export(config)
    onnx_test(config)