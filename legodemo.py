import time
import json 
import json5
import copy
import ast
from scipy.ndimage import binary_dilation, gaussian_filter
import cv2
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
from vllm.lora.request import LoRARequest

from vllm import LLM, SamplingParams

import gradio as gr
import gc
from transformers import pipeline
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import tempfile

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import model_pipeline.all_pipeline.mirmbg.infer_onnx as mirmbg_infer_onnx
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter, ImageStat

import sys
import os
import PIL
import re
from skimage import measure

from scipy import ndimage

from io import BytesIO
from tqdm import tqdm
import glob
import gc

from pymatting import *
from scipy import ndimage

import logging
import traceback
import yaml
from io import BytesIO
from tqdm import tqdm

from torchvision import transforms
from huggingface_hub import hf_hub_download

from model_sam.evf_sam import EvfSamModel
from transformers import AutoTokenizer
from inference_sam import sam_preprocess, beit3_preprocess

from datetime import datetime
import onnxruntime as ort
from skimage.color import rgb2yuv,yuv2rgb
import math

from transformers import CLIPVisionModelWithProjection
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms.functional as tf
import gc

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import model_pipeline.all_pipeline.lama.lama_infer as lama
from model_sam.evf_efficientvit_boxembd import EvfSamEffiVitModel

import random
from torchvision import transforms
from scipy import ndimage
from math import ceil
import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_custom_sampler
from nodes import LoraLoader
from nodes import LoraLoaderModelOnly
from nodes import CLIPTextEncode
from comfy_extras.nodes_ip2p import InstructPixToPixConditioning
from comfy_extras.nodes_flux import CLIPTextEncodeFlux
from comfy_extras import nodes_flux
from comfy_extras import nodes_differential_diffusion
from custom_nodes.comfyui_controlnet_aux import AIO_Preprocessor

AIO_Preprocessor=AIO_Preprocessor()
preprocessor='AnyLineArtPreprocessor_aux'
cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()

CLIPTextEncodeFlux=CLIPTextEncodeFlux()

Inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
Differentialdiffusion = nodes_differential_diffusion.NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
Ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
Vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
from nodes import init_extra_nodes
def import_custom_nodes() -> None:
    init_extra_nodes()
import_custom_nodes()

teacacheforimggen = NODE_CLASS_MAPPINGS["TeaCacheForImgGen"]()
InstructP = InstructPixToPixConditioning()
zerocfg = NODE_CLASS_MAPPINGS["CFGZeroStarAndInit"]() 

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

# 类型定义
ImageType = np.ndarray  # cv2图像实际上是numpy数组
DataObject = Union[ImageType, str, float]


def fix_json_structure(input_json):
    data = input_json
    if "result" in data and isinstance(data["pipeline"], list) and len(data["pipeline"]) > 0:
        data["pipeline"].append({"result":data["result"]})
        del data["result"]
    return data #json.dumps(data, indent=4, ensure_ascii=False)

def process_image(image):
    if image.shape[2] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
        white_background = np.ones_like(rgb) * 255
        processed_image = np.where(alpha[..., np.newaxis] == 0, white_background, rgb)
    elif image.shape[2] == 3:
        processed_image = image
    else:
        raise ValueError("输入的图像通道数必须是3（RGB）或4（RGBA）")
    return processed_image


def smooth_mask_edges(mask, width=3):
    kernel = np.ones((2*width+1, 2*width+1))  
    dilated = binary_dilation(mask, structure=kernel)
    edge_region = dilated ^ mask 
    blurred = gaussian_filter(mask.astype(float), sigma=1)
    result = np.where(edge_region, blurred, mask)
    return result


def move_to_mask(rgba_cv2, input_mask_pil):
    mask = np.array(input_mask_pil)
    mask = (mask > 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgba_cv2.copy()
    target_bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
    tx, ty, tw, th = target_bbox
    target_bottom_center = (tx + tw // 2, ty + th)
    alpha = rgba_cv2[:, :, 3]
    dog_mask = (alpha > 0).astype(np.uint8) * 255
    dog_contours, _ = cv2.findContours(dog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not dog_contours:
        return rgba_cv2.copy()
    dog_bbox = cv2.boundingRect(max(dog_contours, key=cv2.contourArea))
    dx, dy, dw, dh = dog_bbox
    dog_bottom_center = (dx + dw // 2, dy + dh)
    shift_x = target_bottom_center[0] - dog_bottom_center[0]
    shift_y = target_bottom_center[1] - dog_bottom_center[1]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    rows, cols = rgba_cv2.shape[:2]
    translated = cv2.warpAffine(rgba_cv2, M, (cols, rows))
    new_image = np.zeros_like(rgba_cv2)
    new_alpha = translated[:, :, 3]
    new_image[new_alpha > 0] = translated[new_alpha > 0]
    return new_image

def overlay_images1(a, b):
    if b.shape[2] == 4:
        b_rgb = b[:, :, :3]
        b_alpha = b[:, :, 3] / 255.0  
        b_alpha = np.expand_dims(b_alpha, axis=2)
        c = a * (1 - b_alpha) + b_rgb * b_alpha
        c = np.clip(c, 0, 255).astype(np.uint8)
    else:
        c = b
    return c


def fix_broken_json(broken_str):
    fixed_str = re.sub(
        r'{\s*[^"{}\s][^"}]*"\s*,',  # 匹配 { 后面的非法键值
        '{ ',  # 替换为干净的 {
        broken_str
    )
    try:
        data = json.loads(fixed_str)
        return json.dumps(data, indent=2)  # 返回格式化 JSON
    except json.JSONDecodeError as e:
        return None


def modify_json_mask(data):
    # 判断输入类型，如果是字符串就先解析为dict
    input_is_str = isinstance(data, str)
    if input_is_str:
        json_data = json.loads(data)
    else:
        json_data = data
    has_flux_inpaint = any(
        step.get('model') == 'FLUX-INPAINT' 
        for step in json_data.get('pipeline', [])
        if isinstance(step, dict)
    )
    if not has_flux_inpaint:
        for step in json_data.get('pipeline', []):
            if isinstance(step, dict) and step.get('model') == 'ADD-PRED':
                inputs = step.get('input', {})
                if isinstance(inputs, dict) and inputs.get('mask') == 'step1[mask]':
                    inputs['mask'] = None
    return json.dumps(json_data, ensure_ascii=False) if input_is_str else json_data



def filter_a_str(a_str, str_all):
    sub_steps = [step.strip() for step in a_str.split(',')]
    filtered_steps = [step for step in sub_steps if step in str_all]
    return ', '.join(filtered_steps)


def calculate_expansion_ratios(input_mask_PIL):
    mask_array = np.array(input_mask_PIL)
    height, width = mask_array.shape
    
    # 找到黑色区域(值为0)的边界
    rows = np.any(mask_array == 0, axis=1)
    cols = np.any(mask_array == 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return (1.0, 1.0, 1.0, 1.0)  # 如果没有黑色区域，默认各边扩展1倍
    
    # 找到黑色区域的边界
    top = np.argmax(rows)
    bottom = height - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = width - np.argmax(cols[::-1])
    
    # 计算各边需要扩展的比例
    left_ratio = left / (right - left) if (right - left) > 0 else 1.0
    right_ratio = (width - right) / (right - left) if (right - left) > 0 else 1.0
    top_ratio = top / (bottom - top) if (bottom - top) > 0 else 1.0
    bottom_ratio = (height - bottom) / (bottom - top) if (bottom - top) > 0 else 1.0
    
    # 确保最小扩展比例为1.0
    left_ratio = max(1.0, left_ratio)
    right_ratio = max(1.0, right_ratio)
    top_ratio = max(1.0, top_ratio)
    bottom_ratio = max(1.0, bottom_ratio)
    
    return (left_ratio, right_ratio, top_ratio, bottom_ratio)


def prepare_expansion(input_img_PIL, input_mask_PIL):
    # 计算各边需要扩展的比例
    left_ratio, right_ratio, top_ratio, bottom_ratio = calculate_expansion_ratios(input_mask_PIL)
    desired_ratios = (left_ratio, right_ratio, top_ratio, bottom_ratio)
    
    # 获取原始尺寸
    original_width, original_height = input_img_PIL.size
    original_size = (original_width, original_height)
    
    # 计算各边最大扩展比例
    max_left = max(left_ratio, right_ratio)
    max_right = max_left
    max_top = max(top_ratio, bottom_ratio)
    max_bottom = max_top
    max_ratios = (max_left, max_right, max_top, max_bottom)
    
    # 计算新尺寸
    new_width = math.ceil(original_width * max_left + original_width * max_right)
    new_height = math.ceil(original_height * max_top + original_height * max_bottom)
    
    # 创建新图像和mask
    adjusted_img = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    adjusted_mask = Image.new('L', (new_width, new_height), 255)  # 白色背景
    
    # 计算原始图像在新图像中的位置
    paste_x = math.ceil((new_width - original_width) / 2)
    paste_y = math.ceil((new_height - original_height) / 2)
    
    # 粘贴原始图像和mask
    adjusted_img.paste(input_img_PIL, (paste_x, paste_y))
    adjusted_mask.paste(input_mask_PIL, (paste_x, paste_y))
    
    return adjusted_img, adjusted_mask, original_size, max_ratios, desired_ratios



def pad_to_universal_symmetric(input_img, input_mask):
    mask_np = np.array(input_mask.convert('L'))
    h_full, w_full = mask_np.shape

    # 找到原图区域 bbox
    rows = np.any(mask_np == 0, axis=1)
    cols = np.any(mask_np == 0, axis=0)
    top    = np.argmax(rows)
    bottom = h_full - 1 - np.argmax(rows[::-1])
    left   = np.argmax(cols)
    right  = w_full - 1 - np.argmax(cols[::-1])

    orig_w = right - left + 1
    orig_h = bottom - top + 1

    # 四边实际扩展量（像素）
    left_exp   = left
    right_exp  = w_full - 1 - right
    top_exp    = top
    bottom_exp = h_full - 1 - bottom

    # 取四个方向的最大扩展量
    max_exp = max(left_exp, right_exp, top_exp, bottom_exp)

    # 新画布尺寸
    new_w = orig_w + 2 * max_exp
    new_h = orig_h + 2 * max_exp

    # 创建画布
    canvas_img  = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas_mask = Image.new("L",   (new_w, new_h), 255)

    # 把原图区域裁出来，粘到 (max_exp, max_exp)
    crop_box = (left, top, right + 1, bottom + 1)
    orig_img  = input_img.crop(crop_box)
    orig_mask = input_mask.crop(crop_box)
    canvas_img .paste(orig_img,  (max_exp, max_exp))
    canvas_mask.paste(orig_mask, (max_exp, max_exp))

    pad_info = {
        "orig_w": orig_w, "orig_h": orig_h,
        "left_exp": left_exp,   "right_exp": right_exp,
        "top_exp": top_exp,     "bottom_exp": bottom_exp,
        "max_exp": max_exp,
    }
    return canvas_img, canvas_mask, pad_info

def crop_to_asymmetric(input_img, pad_info):
    me = pad_info["max_exp"]
    lx = pad_info["left_exp"]
    rx = pad_info["right_exp"]
    ty = pad_info["top_exp"]
    by = pad_info["bottom_exp"]
    ow = pad_info["orig_w"]
    oh = pad_info["orig_h"]
    x0 = me - lx
    y0 = me - ty
    fw = ow + lx + rx
    fh = oh + ty + by
    return input_img.crop((x0, y0, x0 + fw, y0 + fh))



def has_black_borders_on_all_sides(image, threshold=10):
    img_array = np.array(image)
    return (np.all(img_array[0, :] <= threshold) and 
            np.all(img_array[-1, :] <= threshold) and
            np.all(img_array[:, 0] <= threshold) and 
            np.all(img_array[:, -1] <= threshold))


def overlay_transparent(image, mask, square_size=20):
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)  # 转为 RGBA
    image_rgba[:, :, 3] = mask
    return image_rgba

def get_bbox_from_mask(mask):
    y_coords, x_coords = np.where(mask == 255)
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    return [x_min, y_min, x_max, y_max]


def align_and_calculate_overlap(bbox1, bbox2):
    # 获取bbox1和bbox2的坐标
    x_min1, y_min1, x_max1, y_max1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x_min2, y_min2, x_max2, y_max2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    print(x_min1, y_min1, x_max1, y_max1)
    print(x_min2, y_min2, x_max2, y_max2)

    # 计算bbox1和bbox2的x轴中心
    x_center1 = (x_min1 + x_max1) / 2
    x_center2 = (x_min2 + x_max2) / 2

    # 计算x平移距离
    x_translation = x_center1 - x_center2

    # 平移bbox2使得中心点对齐
    x_min2 += x_translation
    x_max2 += x_translation

    # 计算y平移距离，使得下边对齐
    y_translation = y_max1 - y_max2

    y_min2 += y_translation
    y_max2 += y_translation
    print(x_min2, y_min2, x_max2, y_max2)

    # 计算重叠区域的坐标
    overlap_x_min = max(x_min1, x_min2)
    overlap_y_min = max(y_min1, y_min2)
    overlap_x_max = min(x_max1, x_max2)
    overlap_y_max = min(y_max1, y_max2)

    # 确保存在重叠区域
    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
        print('res', overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max)
        return [overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max]
    else:
        return None  # 如果没有重叠区域，返回None


def shift_mask_based_on_center(mask, obj_mask, direction):
    a, b = mask.shape
    msk = mask - obj_mask
    msk[msk < 0] = 0
    sum_all = np.sum(msk)
    y_coords, x_coords = np.where(mask == 255)
    center_x = np.mean(x_coords)
    if center_x < b / 2:
        shift_direction = 'left'
        shift_direction_in = 'right'
    else:
        shift_direction = 'right'
        shift_direction_in = 'left'
    s = 0.025
    shifted_mask_tmp = msk.copy()
    if direction:
        for k in range(20):
            shift_amount = int(b * s)
            s += 0.025
            shifted_mask = np.zeros_like(mask)
            if shift_direction == 'left':
                shifted_mask[:, :b-shift_amount] = mask[:, shift_amount:]
            else:
                shifted_mask[:, shift_amount:] = mask[:, :b-shift_amount]
            shifted_mask_noc = shifted_mask - obj_mask
            shifted_mask_noc[shifted_mask_noc < 0] = 0
            if np.sum(shifted_mask_noc) > sum_all:
                sum_all = np.sum(shifted_mask_noc)
                shifted_mask_tmp = shifted_mask
            if np.sum(shifted_mask) != np.sum(mask):
                break
        return shifted_mask_tmp 
    else:
        shifted_mask_tmp_in = msk.copy()
        for k in range(20):
            shift_amount = int(b * s)
            s += 0.025
            shifted_mask = np.zeros_like(mask)
            shifted_mask[:, :b-shift_amount] = mask[:, shift_amount:]
            shifted_mask_noc = shifted_mask - obj_mask
            shifted_mask_noc[shifted_mask_noc < 0] = 0
            if np.sum(shifted_mask_noc) > sum_all:
                sum_all = np.sum(shifted_mask_noc)
                shifted_mask_tmp_in = shifted_mask
        if np.sum(shifted_mask_tmp_in) > np.sum(shifted_mask_tmp):
            shifted_mask_tmp1 = shifted_mask_tmp_in
        else:
            shifted_mask_tmp1 = shifted_mask_tmp

        shifted_mask_tmp_in = msk.copy()
        s = 0.025
        for k in range(20):
            shift_amount = int(b * s)
            s += 0.025
            shifted_mask = np.zeros_like(mask)
            shifted_mask[:, shift_amount:] = mask[:, :b-shift_amount]
            shifted_mask_noc = shifted_mask - obj_mask
            shifted_mask_noc[shifted_mask_noc < 0] = 0
            if np.sum(shifted_mask_noc) > sum_all:
                sum_all = np.sum(shifted_mask_noc)
                shifted_mask_tmp_in = shifted_mask
        if np.sum(shifted_mask_tmp_in) > np.sum(shifted_mask_tmp):
            shifted_mask_tmp2 = shifted_mask_tmp_in
        else:
            shifted_mask_tmp2 = shifted_mask_tmp

        if np.sum(shifted_mask_tmp1) > np.sum(shifted_mask_tmp2):
            return shifted_mask_tmp1
        else:
            return shifted_mask_tmp2



def draw_add_mask(image_size, bbox):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask

def preprocess_np_to_torch(np_array, target_shape=(1, 682, 1024, 3)):
    if len(np_array.shape) == 2:
        np_array = np.stack((np_array,) * 3, axis=-1)
    elif len(np_array.shape) == 3 and np_array.shape[2] != 3:
        raise ValueError(f"不支持的输入形状: {np_array.shape}，必须是 (height, width) 或 (height, width, 3)")
    if np_array.min() < 0 or np_array.max() > 255:
        raise ValueError("输入数组的值应该在 0 到 255 之间")
    np_normalized = np_array.astype(np.float32) / 255.0
    from skimage.transform import resize
    resized_array = resize(np_normalized, (target_shape[1], target_shape[2], target_shape[3]), mode='reflect', preserve_range=True)
    tensor = torch.from_numpy(resized_array).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = tensor[:,0:1,:,:]
    return tensor

def overlay_images_with_mask(image1, image2, mask):
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    image1_masked = Image.composite(image1, Image.new("RGB", image1.size, (0, 0, 0)), mask_pil)
    mask_inv_pil = Image.fromarray((255 - mask).astype(np.uint8))  # 反掩码
    image2_masked = Image.composite(image2, Image.new("RGB", image2.size, (0, 0, 0)), mask_inv_pil)
    result = ImageChops.add(image1_masked, image2_masked)
    return result



def create_custom_outpainting_mask_and_image(image, left_ratio, right_ratio, top_ratio, bottom_ratio):
    H, W = image.shape[:2]
    left_ext = int(W * (left_ratio - 1))
    right_ext = int(W * (right_ratio - 1))
    top_ext = int(H * (top_ratio - 1))
    bottom_ext = int(H * (bottom_ratio - 1))
    new_W = W + left_ext + right_ext
    new_H = H + top_ext + bottom_ext
    mask = np.zeros((new_H, new_W), dtype=np.uint8)
    mask[:] = 255  # 全白
    x1, y1 = left_ext, top_ext
    x2, y2 = x1 + W, y1 + H
    mask[y1:y2, x1:x2] = 0
    if len(image.shape) == 3:  # 彩色图
        padded_img = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    else:  # 灰度图
        padded_img = np.zeros((new_H, new_W), dtype=np.uint8)
    padded_img[y1:y2, x1:x2] = image
    long_side = 1024
    scale = long_side / max(new_W, new_H)
    temp_W = int(new_W * scale)
    temp_H = int(new_H * scale)
    def make_divisible(x):
        return max(64, int(round(x / 64) * 64))
    final_W = make_divisible(temp_W)
    final_H = make_divisible(temp_H)
    if new_W > new_H:  # 原图更宽
        final_W = long_side
        final_H = make_divisible(long_side * new_H / new_W)
    else:  # 原图更高
        final_H = long_side
        final_W = make_divisible(long_side * new_W / new_H)
    resized_mask = cv2.resize(mask, (final_W, final_H), interpolation=cv2.INTER_NEAREST)
    resized_img = cv2.resize(padded_img, (final_W, final_H), interpolation=cv2.INTER_AREA)
    return resized_mask, resized_img



def check_quality(lama_res_PIL, mask, threshold=30):
    image_np = np.array(lama_res_PIL.convert('L'))  # 转换为灰度图
    if mask.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    if image_np.shape != mask.shape:
        raise ValueError("图像和 mask 的大小不一致")
    mask_region = image_np[mask == 255]
    mask_mean = np.mean(mask_region) if len(mask_region) > 0 else 0
    non_mask_region = image_np[mask == 0]
    non_mask_mean = np.mean(non_mask_region) if len(non_mask_region) > 0 else 0
    if abs(mask_mean - non_mask_mean) > threshold:
        quality = "质量低"
    else:
        quality = "质量高"
    return abs(mask_mean - non_mask_mean)

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0)  # 从 [1, 3, H, W] 变为 [3, H, W]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.mul(255).byte()
    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()  # 从 [3, H, W] 变为 [H, W, 3]
    pil_image = Image.fromarray(numpy_image)
    return pil_image

def mask_to_pil_l(mask):
    mask = mask.squeeze(0).squeeze(0)  # 从 [1, 1, H, W] 变为 [H, W]
    mask = mask.mul(255).byte()
    numpy_mask = mask.cpu().numpy()
    pil_image = Image.fromarray(numpy_mask, mode='L')
    return pil_image

def compute_canny(canny_image):
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(canny_image)
    tensor[tensor <= 0.5] = 0
    tensor[tensor > 0.5] = 1
    tensor_3ch = torch.cat([tensor] * 3, dim=0)
    tensor_3ch = tensor_3ch.unsqueeze(0)
    canny = tensor_3ch.permute(0, 2, 3, 1)
    return canny

def convert_mask_to_torch(mask):
    height, width = mask.shape
    new_mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    resized_mask = mask.astype(np.float32) / 255.0
    new_mask[0, 0] = torch.from_numpy(resized_mask)
    return new_mask

def convert_image_to_torch(image):
    height, width, _ = image.shape
    new_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    image = image.astype(np.float32) / 255.0
    new_image[0] = torch.from_numpy(image)
    return new_image

def save_torch_tensor_as_image(tensor, size=None):
    tensor = tensor.squeeze(0).detach().cpu().numpy()  
    tensor = (tensor * 255).astype(np.uint8)  
    image = Image.fromarray(tensor)
    if size != None:
        image = image.resize(size)
    return image

class WorkflowExecutor:
    def __init__(self, llm, sampling_params, processor_workflow):
        self.llm = llm
        self.sampling_params = sampling_params
        self.processor_workflow = processor_workflow
        # self.workflow = self.load_workflow(workflow_file)
        # self.context = {}  # 存储所有中间结果
        self.model_mapping = self.initialize_model_mapping()
        _default_session_options = ort.capi._pybind_state.get_default_session_options()
        def get_default_session_options_new():
            _default_session_options.inter_op_num_threads = 1
            _default_session_options.intra_op_num_threads = 1
            return _default_session_options
        ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        self.lora_path = "./mimo_lora/"

        version = "./CVRES"
        self.model = EvfSamEffiVitModel.from_pretrained(version, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(version,padding_side="right",use_fast=False,)
        self.model.to("cuda")
        self.model.eval()
        with torch.inference_mode():
            self.clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
            self.vae = VAELoader.load_vae("ae.safetensors")[0]
            self.unet_fill = UNETLoader.load_unet("flux1-fill-dev.safetensors", "fp8_e4m3fn_fast")[0]
            self.unet_canny = UNETLoader.load_unet("flux1-canny-dev.safetensors", "fp8_e4m3fn_fast")[0]
            self.unet_canny = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=self.unet_canny)[0]
            self.unet_canny = zerocfg.patch(model=self.unet_canny, use_zero_init=True, zero_init_steps=0)[0]
            self.FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()

        self.subjectSeg = mirmbg_infer_onnx.segment_MIRMBG('./CVSOS/') 
        self.lamaInpaint = lama.lama_Inpaint()


    
    def initialize_model_mapping(self) -> Dict[str, Any]:
        """初始化模型函数映射，实际使用时应该替换为你的模型实现"""
        return {
            "CMI-PRED": self.dummy_captionmask_pred,
            "RES": self.dummy_res,
            "MASK-SEG": self.dummy_mask_seg,
            "FASTINPAINT": self.dummy_fastinpaint,
            "FLUX-FILL": self.dummy_flux_fill,
            "FLUX-INPAINT": self.dummy_flux_inpaint,
            "INVERSE": self.dummy_inverse,
            "COMPOSE": self.dummy_compose,
            "RESIZE": self.dummy_resize,
            "BBOX": self.dummy_bbox,
            "SOS": self.dummy_sos,
            "FLUX-CBG": self.dummy_flux_cbg,
            "ADD-PRED": self.dummy_add_pred,
            "FLUX-STYLE": self.dummy_flux_style,
            "FLUX-RCM": self.dummy_flux_rcm,
            "FLUX-ENV": self.dummy_flux_env,
            "FLUX-POSE": self.dummy_flux_pose
            # 添加更多模型映射...
        }

    def dummy_flux_rcm(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        unet_mc = loraloadermodelonly.load_lora_model_only(lora_name="mat768.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_mc = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=unet_mc)[0]
        prompt = inputs["prompt"]
        image_np = inputs["image"].copy()
        image = Image.fromarray(image_np[:,:,::-1])
        ww, hh = image.size[0], image.size[1]
        kernel_size = 35
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        try:
            print('res+rcm')
            mask = inputs["mask"].copy()
            mask_right = cv2.dilate(mask, kernel, iterations=1)
        except:
            mask_right = np.ones((hh, ww), dtype=np.uint8) * 255

        instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'
        
        if image.size[0] != 768:
            new_width = 768
            scale = new_width / image.size[0]
            new_height = int(image.size[1] * scale)
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height))
            mask_right = cv2.resize(mask_right, (new_width, new_height))
        
        width, height = image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(image, (width, 0))
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = mask_right
        combined_image = np.array(combined_image)
        inputMask = convert_mask_to_torch(mask_array)
        inputImage = convert_image_to_torch(combined_image)

        with torch.inference_mode():
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
            cliptextencode_pos = cliptextencode.encode(text=instruction, clip=self.clip)[0]
            fluxguidance = self.FluxGuidance.append(guidance=50, conditioning=cliptextencode_pos)[0]
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            differentialdiffusion = Differentialdiffusion.apply(model=unet_mc)[0]
            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=25,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)

        e_image = e_image.crop((width,0,width*2,height))
        e_image = e_image.resize((ww, hh))
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }

    def dummy_flux_env(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        unet_env = loraloadermodelonly.load_lora_model_only(lora_name="env768.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_env = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=unet_env)[0]

        prompt = inputs["prompt"] 
        instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'
        image_np = inputs["image"].copy()
        image = Image.fromarray(image_np[:,:,::-1])
        ww, hh = image.size[0], image.size[1]
        
        if image.size[0] != 768:
            new_width = 768
            scale = new_width / image.size[0]
            new_height = int(image.size[1] * scale)
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height))
        
        width, height = image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(image, (width, 0))
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = 255
        combined_image = np.array(combined_image)
        inputMask = convert_mask_to_torch(mask_array)
        inputImage = convert_image_to_torch(combined_image)

        with torch.inference_mode():
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
            cliptextencode_pos = cliptextencode.encode(text=instruction, clip=self.clip)[0]
            fluxguidance = self.FluxGuidance.append(guidance=50, conditioning=cliptextencode_pos)[0]
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            differentialdiffusion = Differentialdiffusion.apply(model=unet_env)[0]
            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=25,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)

        e_image = e_image.crop((width,0,width*2,height))
        e_image = e_image.resize((ww, hh))
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }

    def dummy_flux_pose(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        unet_pose = loraloadermodelonly.load_lora_model_only(lora_name="pose768.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_pose = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=unet_pose)[0]

        prompt = inputs["prompt"] 
        instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'
        image_np = inputs["image"].copy()
        image = Image.fromarray(image_np[:,:,::-1])
        ww, hh = image.size[0], image.size[1]
        
        if image.size[0] != 768:
            new_width = 768
            scale = new_width / image.size[0]
            new_height = int(image.size[1] * scale)
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height))
        
        width, height = image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(image, (width, 0))
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = 255
        '''
        right_side = mask_array[:, width:]
        h_center = height // 2
        w_center = width // 2
        h_start = int(height * 0.05)    # 上下各留10%的黑边
        h_end = int(height * 0.95)      # 高度方向中心占80%
        w_start = int(width * 0.05)     # 左右各留10%的黑边
        w_end = int(width * 0.95)       # 宽度方向中心占80%
        right_side[h_start:h_end, w_start:w_end] = 255
        mask_array = right_side
        '''
        combined_image = np.array(combined_image)
        inputMask = convert_mask_to_torch(mask_array)
        inputImage = convert_image_to_torch(combined_image)

        with torch.inference_mode():
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
            cliptextencode_pos = cliptextencode.encode(text=instruction, clip=self.clip)[0]
            fluxguidance = self.FluxGuidance.append(guidance=50, conditioning=cliptextencode_pos)[0]
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            differentialdiffusion = Differentialdiffusion.apply(model=unet_pose)[0]
            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=25,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)

        e_image = e_image.crop((width,0,width*2,height))
        e_image = e_image.resize((ww, hh))
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }
        
    def dummy_add_pred(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        ori_image = inputs['image'].copy()
        ori_image = ori_image[:,:,::-1]
        ori_image = Image.fromarray(ori_image)
        ww, hh = ori_image.size[0], ori_image.size[1]
        size = 768
        if ww > hh:
            rate_wh = hh/ww
            size_w = int(size)
            size_h = int(rate_wh*size)
        elif ww < hh:
            rate_wh = ww/hh
            size_h = int(size)
            size_w = int(rate_wh*size)
        else:
            size_h = int(size)
            size_w = int(size)
        resize_image = ori_image.resize((size_w, size_h))

        ppp = "Add " + inputs["prompt"] + ", predict where it should be placed. Output the answer in the format [x1, y1, x2, y2]"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": resize_image, 
                    },
                    {"type": "text", "text": ppp},
                ],
            }
        ]
        prompt_post = self.processor_workflow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        llm_inputs = {
            'prompt': prompt_post,
            'multi_modal_data': mm_data,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=LoRARequest("lora_name", 1, self.lora_path))
        text = outputs[0].outputs[0].text
        text = text.split('</think>')[-1]
        print('text', text)
        bbox = ast.literal_eval(text)
        image_size = resize_image.size
        mask = draw_add_mask(image_size, bbox)
        
        shift_pixels = int(hh/48)
        if 'hat' in inputs['prompt'] or 'cap' in inputs['prompt']:
            mask = mask.transform(
                mask.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, shift_pixels),  # 平移参数 (a, b, c, d, e, f)
                fillcolor=0  # 填充平移后空白区域（根据你的mask类型调整，0通常是黑色）
            )
        mask = np.array(mask)
        ratio = np.sum(mask)/(255*np.sum(np.ones_like(mask)))
        if ratio < 0.002:
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=5)

        print('ratio', np.sum(mask)/(255*np.sum(np.ones_like(mask))))


        if inputs['mask'] is None:
            return {"mask": mask}
        else:
            mask_add = mask.copy()
            mask_ori = inputs['mask'].copy()
            bbox_ori = get_bbox_from_mask(mask_ori)
            bbox_add = get_bbox_from_mask(mask_add)
            bbox3 = align_and_calculate_overlap(bbox_ori, bbox_add)
            width, height = mask_ori.shape[1], mask_ori.shape[0]
            black_mask = Image.new('L', (width, height), color=0)
            left_top = (int(bbox3[0]), int(bbox3[1]))  # 左上角点 (x1, y1)
            right_bottom = (int(bbox3[2]), int(bbox3[3]))  # 右下角点 (x2, y2)
            draw = ImageDraw.Draw(black_mask)
            draw.rectangle([left_top, right_bottom], fill=255)
            mask_add = np.array(black_mask)
            return {"mask": mask_add}

    def dummy_flux_style(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:

        unet_style = loraloadermodelonly.load_lora_model_only(lora_name="style768.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_style = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=unet_style)[0]

        prompt1 = inputs["prompt"] 
        style1 = inputs["style"]
        describe = prompt1

        if style1 is not None:
            if 'style' in style1:
                prompt = style1
            else:
                prompt = style1 + ' style'
        else:
            prompt = "anime style"

        if 'ink' in prompt and 'hinese' not in prompt:
            prompt = prompt.replace('ink', 'Chinese ink')


        prompt = "Convert the image into " + prompt + "." #+ ' ' + describe 
        instruction = 'A diptych with two side-by-side images of the same scene. ' + describe + 'On the right, the scene is exactly the same as on the left but ' + prompt
     #   instruction = 'A diptych with two side-by-side images of the same scene. ' + 'On the right, the scene is exactly the same as on the left but ' + prompt
        print(instruction)
        image_np = inputs["image"].copy()
        image = Image.fromarray(image_np[:,:,::-1])
        ww, hh = image.size[0], image.size[1]

        try:
            print('res+style')
            mask_right = inputs["mask"].copy()            
        except:
            mask_right = np.ones((hh, ww), dtype=np.uint8) * 255
        
        if image.size[0] != 768:
            new_width = 768
            scale = new_width / image.size[0]
            new_height = int(image.size[1] * scale)
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height))
            mask_right = cv2.resize(mask_right, (new_width, new_height))
        
        width, height = image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(image, (width, 0))
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = mask_right
        combined_image = np.array(combined_image)
        inputMask = convert_mask_to_torch(mask_array)
        inputImage = convert_image_to_torch(combined_image)

        with torch.inference_mode():
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
            cliptextencode_pos = cliptextencode.encode(text=instruction, clip=self.clip)[0]
            fluxguidance = self.FluxGuidance.append(guidance=50, conditioning=cliptextencode_pos)[0]
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            differentialdiffusion = Differentialdiffusion.apply(model=unet_style)[0]
            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=25,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)

        e_image = e_image.crop((width,0,width*2,height))
        e_image = e_image.resize((ww, hh))
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }


    def dummy_flux_cbg(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        src_height, src_width = inputs['image'].shape[:2]
        image_ori = inputs['image'].copy()
        mask_pil_inv = inputs['mask'].copy()
        image_pil = inputs['image'].copy()
        image_PIL = Image.fromarray(image_pil[:,:,::-1])
        mask_pil = 255 - mask_pil_inv.copy()
        mm_ori = mask_pil.copy()
        mask_pil[mask_pil < 127.5] = 0
        mask_pil[mask_pil >= 127.5] = 255

        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dil = cv2.dilate(mask_pil.copy(), kernel, iterations=1)

        mask = np.where(mask_pil>127, 255, 0)
        src_mask = mask.reshape(mask.shape[0], mask.shape[1]).astype(np.uint8)
        add_value = 2
        bottom_point = np.max(np.where(np.any(mask == 255, axis=1))) + add_value
        top_point = np.min(np.where(np.any(mask == 255, axis=1))) - add_value
        left_point = np.min(np.where(np.any(mask == 255, axis=0))) - add_value
        right_point = np.max(np.where(np.any(mask == 255, axis=0))) + add_value

        mask_pil = mask_pil//255
        mask_expanded = np.expand_dims(mask_pil, axis=-1)
        mask_expanded = np.broadcast_to(mask_expanded, image_pil.shape)
        masked_image = image_pil * mask_expanded
        pil_image = Image.fromarray(masked_image[:,:,::-1])

        ww, hh = pil_image.size[0], pil_image.size[1]
        size = 768 
        if ww > hh:
            rate_wh = hh/ww
            size_w = int(size)
            size_h = int(rate_wh*size)
        elif ww < hh:
            rate_wh = ww/hh
            size_h = int(size)
            size_w = int(rate_wh*size)
        else:
            size_h = int(size)
            size_w = int(size)
        resize_image = pil_image.resize((size_w, size_h))

        scene = inputs['prompt']
        print('scene', scene)

        ppp = inputs["prompt"] + ", give a description of the image after changing the background"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": resize_image, 
                    },
                    {"type": "text", "text": ppp},
                ],
            }
        ]
        prompt_post = self.processor_workflow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        llm_inputs = {
            'prompt': prompt_post,
            'multi_modal_data': mm_data,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=LoRARequest("lora_name", 1, self.lora_path))
        text = outputs[0].outputs[0].text
        text = text.split('</think>')[-1]
        print('text', text)
        prompt = scene + ', ' + text.replace('\n','')
        print('prompt', prompt)
        with torch.inference_mode():
            differentialdiffusion = Differentialdiffusion.apply(model=self.unet_canny)[0]
            cliptextencode_pos = cliptextencode.encode(text=prompt, clip=self.clip,)[0]
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip,)[0]
            fluxguidance = self.FluxGuidance.append(guidance=30, conditioning=cliptextencode_pos)[0]
            VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            image = loadimage.load_image(image=image_PIL)[0]
            latent_image = VAEEncode.encode(self.vae, image)[0]
        mask_matting = preprocess_np_to_torch(mask_pil_inv)
        refine_step = 3
        for r in range(refine_step):
            image = loadimage.load_image(image=image_PIL)[0]
            with torch.inference_mode():
                if r == 0:
                    cond_image = image_PIL * (mask.reshape(mask.shape[0], mask.shape[1], 1) > 128)
                    cond_image = Image.fromarray(cond_image)
                    cropped_image = cond_image.crop((left_point, top_point, right_point, bottom_point))
                    cv_image = np.array(cropped_image.convert('RGB'))
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    edges = cv2.Canny(cv_image, 100, 200)
                    edges_pil = Image.fromarray(edges)
                    black_background = Image.new('RGB', cond_image.size, color='black')
                    black_background.paste(edges_pil, (left_point, top_point), mask=edges_pil)
                    canny_image = black_background.convert('L')
                    canny = compute_canny(canny_image)
                # anyline canny
                else:
                    short = int(min(image.shape[1], image.shape[2]))
                    canny_condtion = AIO_Preprocessor.execute(preprocessor, image, resolution=short)
                    canny = canny_condtion[0]
                    canny_array = canny.squeeze(0).numpy()
                    canny_array = (canny_array * 255).astype(np.uint8)
                    canny_image_model = Image.fromarray(canny_array)
                    cond_image = image_PIL * (mask.reshape(mask.shape[0], mask.shape[1], 1) > 127.5)
                    cond_image = Image.fromarray(cond_image)
                    cropped_image = cond_image.crop((left_point, top_point, right_point, bottom_point))
                    cv_image = np.array(cropped_image.convert('RGB'))
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    edges = cv2.Canny(cv_image, 100, 200)
                    edges_pil = Image.fromarray(edges)
                    black_background = Image.new('RGB', cond_image.size, color='black')
                    black_background.paste(edges_pil, (left_point, top_point))
                    black_background = black_background.convert('L')
                    canny_image = overlay_images_with_mask(black_background, canny_image_model, mask_dil)
                    canny_image = canny_image.convert('L')
                    canny = compute_canny(canny_image)

                inpaintmodelconditioning = InstructP.encode(
                    positive=fluxguidance,
                    negative=cliptextencode_neg,
                    vae=self.vae,
                    pixels=canny,
                )
                inpaintmodelconditioning[2]['noise_mask'] = mask_matting #torch.ones_like(mask_matting) #mask_matting 
                inpaintmodelconditioning[2]['samples'] = latent_image['samples']
                if r == refine_step-1:
                    steps = 20
                else:
                    steps = 16
                ksampler = Ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=steps, #20,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="normal",
                    denoise=0.95, #1,
                    model=differentialdiffusion,
                    positive=inpaintmodelconditioning[0],
                    negative=inpaintmodelconditioning[1],
                    latent_image=inpaintmodelconditioning[2],
                )
                vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
                save_img = vaedecode[0]
                image_bgchanged = save_torch_tensor_as_image(save_img)
                # lama
                image_bgchanged = np.array(image_bgchanged)
                image_bgchanged_beforelama = image_bgchanged
                mask_ori = mask.copy() #subjectMask0.copy()
                mask_sd = self.subjectSeg.forward(image_bgchanged_beforelama)
                image_bgchanged = self.lamaInpaint.forward(image_bgchanged_beforelama, mask_ori, mask_sd, True)
                ## sr
                image_bgchanged = cv2.resize(image_bgchanged, (src_width, src_height))
                image_bgchanged_ori = image_bgchanged.copy()
                mask_gaussianblured = mm_ori/255
                mask_gaussianblured = np.stack((mask_gaussianblured, ) * 3, axis=-1)
                image_bgchanged = image_bgchanged[:,:,::-1]
                image_bgchanged = image_bgchanged * (1 - mask_gaussianblured) + image_ori * mask_gaussianblured
                image_bgchanged = image_bgchanged.astype(np.uint8)
                if r == refine_step-1:
                    return {
                        "image": image_bgchanged
                    }
                else:
                    image_PIL = Image.fromarray(image_bgchanged[:,:,::-1])


    def dummy_inverse(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        # mask
        if inputs['mask2'] is not None:
            try:
                mask_a = inputs['mask1'].copy()
                mask_b = inputs['mask2'].copy()
                mask = mask_a - mask_b
                mask[mask < 0] = 0
            except:
                mask_b = inputs['mask2'].copy()
                mask = 255 - mask_b
            return {
                "mask": mask,
                "image": None
            }
        else:
            try:
                img_a = inputs['image1'].copy()
                img_b = inputs['image2'].copy()
                img = img_a - img_b
                img[img < 0] = 0
            except:
                img = None
            return {
                "mask": None,
                "image": img
            }


    def dummy_compose(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        # image
        if inputs['image2'] is not None:
            image_a = inputs['image1'].copy()
            image_b = inputs['image2'].copy()
            image = overlay_images1(image_a, image_b)
            return {
                "mask": None,
                "image": image
            }
        else:
            mask_a = inputs['mask1'].copy()
            mask_b = inputs['mask2'].copy()
            mask = mask_a + mask_b
            return {
                "mask": mask,
                "image": None
            }


    def dummy_resize(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        scale_factor = inputs['ratio']
        if inputs['mask'] is not None:
            mask = inputs['mask'].copy()
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return mask
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            object_region = mask[y:y+h, x:x+w]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            resized_object = cv2.resize(object_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            new_mask = np.zeros_like(mask)
            new_x = x + (w - new_w) // 2  
            new_y = y + h - new_h        
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            end_y = min(new_y + new_h, mask.shape[0])
            end_x = min(new_x + new_w, mask.shape[1])
            resized_object = resized_object[:end_y-new_y, :end_x-new_x]
            new_mask[new_y:end_y, new_x:end_x] = resized_object
            return {
                "mask": new_mask,
                "image": None
            }
        else:
            image = inputs['image'].copy()
            print('iiiiiiiiiiiiiiiiiiii', image.shape)
            if image.shape[2] == 4:
                b, g, r, a = cv2.split(image)
                _, mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return image
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                object_region = image[y:y+h, x:x+w]
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR
                resized_object = cv2.resize(object_region, (new_w, new_h), interpolation=interpolation)
                output_image = np.zeros_like(image)
                new_x = x + (w - new_w) // 2  # 水平居中
                new_y = y + h - new_h         # 底边对齐
                new_x = max(0, new_x)
                new_y = max(0, new_y)
                end_y = min(new_y + new_h, image.shape[0])
                end_x = min(new_x + new_w, image.shape[1])
                resized_h = end_y - new_y
                resized_w = end_x - new_x
                resized_object = resized_object[:resized_h, :resized_w]
                output_image[new_y:end_y, new_x:end_x] = resized_object
            else:
                h, w = image.shape[:2]
                zoomed = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                zoomed_h, zoomed_w = zoomed.shape[:2]
                start_x = (zoomed_w - w) // 2
                start_y = (zoomed_h - h) // 2
                output_image = zoomed[start_y:start_y + h, start_x:start_x + w]

            return {
                "mask": None,
                "image": output_image
            }



    def dummy_bbox(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        input_mask = inputs['mask'].copy()
        '''
        if len(input_mask.shape) > 2:
            input_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_mask = np.zeros_like(binary_mask)
        if contours:
            all_points = np.vstack(contours)
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.fillPoly(output_mask, [box], 255)
            '''
        rows = np.any(input_mask == 255, axis=1)
        cols = np.any(input_mask == 255, axis=0)
        if not np.any(rows) or not np.any(cols):
            bbox_mask = np.zeros_like(input_mask)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_mask = np.zeros_like(input_mask)
        bbox_mask[rmin:rmax+1, cmin:cmax+1] = 255
        output_mask = bbox_mask
        return {"mask": output_mask}
        

    def dummy_sos(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        image_ori = inputs['image'].copy()
        src_height, src_width = inputs['image'].shape[:2]
        image_np = inputs['image'].copy()
        subjectMask = self.subjectSeg.forward(image_np)
        subjectMask = np.where(subjectMask>127, 255, 0)
        subjectMask = subjectMask.astype(np.uint8)
        subjectMask = cv2.resize(subjectMask, (src_width, src_height))
        mask_image = cv2.resize(subjectMask, (src_width, src_height))
        img_result = overlay_transparent(image_ori, mask_image)
        return {
            "mask": mask_image,
            "image": img_result
        }

    def dummy_mask_seg(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        image_ori = inputs['image'].copy()
        mask_ori = inputs['mask'].copy()
        # 1. 裁剪图像，并记录 bbox
        mask = (mask_ori == 255)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, mask.shape[0] - 1)
        cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, mask.shape[1] - 1)
        cropped_image = image_ori[rmin:rmax+1, cmin:cmax+1, :]

        # 2. 主体分割 & matting 处理（原逻辑）
        src_height, src_width = cropped_image.shape[:2]
        image_np = cropped_image.copy()
        subjectMask = self.subjectSeg.forward(image_np)
        subjectMask = np.where(subjectMask > 127, 255, 0).astype(np.uint8)
        subjectMask = cv2.resize(subjectMask, (src_width, src_height))

        image_path = cropped_image.copy()
        alpha = subjectMask
        mask_image = cv2.resize(alpha, (src_width, src_height))
        img_result = overlay_transparent(cropped_image, mask_image)  # 裁剪图的透明叠加
        full_result_rgba = np.zeros((image_ori.shape[0], image_ori.shape[1], 4), dtype=np.uint8)
        full_result_rgba[rmin:rmax+1, cmin:cmax+1] = img_result

        full_mask = np.zeros_like(image_ori[:, :, 0], dtype=np.uint8)  # 单通道 mask
        full_mask[rmin:rmax+1, cmin:cmax+1] = mask_image  # 贴回 mask

        return {
            "mask": full_mask,  # 原图尺寸的 mask
            "image": full_result_rgba  # 原图尺寸的透明叠加结果
        }


    # 以下是示例模型函数，实际使用时应该替换为你的真实模型实现
    def dummy_captionmask_pred(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        ori_image = inputs['image'].copy()
        ori_image = ori_image[:,:,::-1]
        ori_image = Image.fromarray(ori_image)
        ww, hh = ori_image.size[0], ori_image.size[1]

        size = 448 #1024
        if ww > hh:
            rate_wh = hh/ww
            size_w = int(size)
            size_h = int(rate_wh*size)
        elif ww < hh:
            rate_wh = ww/hh
            size_h = int(size)
            size_w = int(rate_wh*size)
        else:
            size_h = int(size)
            size_w = int(size)
        resize_image = ori_image.resize((size_w, size_h))

        ppp = "Describe this image"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": resize_image, 
                    },
                    {"type": "text", "text": ppp},
                ],
            }
        ]
        prompt_post = self.processor_workflow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        llm_inputs = {
            'prompt': prompt_post,
            'multi_modal_data': mm_data,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, lora_request=LoRARequest("lora_name", 1, self.lora_path))
        text = outputs[0].outputs[0].text
        text = text.split('</think>')[-1]

        caption = text.replace('\n','')
        if 'left_ratio' in inputs.keys() and inputs['left_ratio'] is None:
            mask = None
            resized_image = None
        elif 'ratio' in inputs.keys() and inputs['ratio'] is None:
            mask = None
            resized_image = None
        else:
            left_ratio = inputs['left_ratio']
            right_ratio = inputs['right_ratio']
            top_ratio = inputs['top_ratio']
            bottom_ratio = inputs['bottom_ratio']
            # 全图扩展
            if (left_ratio == right_ratio == top_ratio == bottom_ratio) and left_ratio != 1:
                left_ratio = 0.5*left_ratio + 0.5
                right_ratio = 0.5*right_ratio + 0.5
                top_ratio = 0.5*top_ratio + 0.5
                bottom_ratio = 0.5*bottom_ratio + 0.5

            mask, resized_image = create_custom_outpainting_mask_and_image(inputs['image'], left_ratio, right_ratio, top_ratio, bottom_ratio)
        #    mask, resized_image = create_custom_outpainting_mask_and_image(inputs['image'], 3,1,1,1)

        return {
            "caption": caption,
            "image": resized_image,
            "mask": mask
        }

    
    def dummy_res(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        image_ori = inputs['image'].copy()
        image_np = inputs['image'].copy()
        prompt = inputs['prompt']
        original_size_list = [image_np.shape[:2]]
        image_beit = beit3_preprocess(image_np, 224).to(dtype=self.model.dtype, device=self.model.device)
        image_sam, resize_shape = sam_preprocess(image_np, model_type="ori")
        image_sam = image_sam.to(dtype=self.model.dtype, device=self.model.device)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=self.model.device)
        pred_mask = self.model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        pred_mask = pred_mask * 255
        pred_mask = pred_mask.astype(np.uint8)

        img_result = overlay_transparent(image_ori, pred_mask)
        return {
            "mask": pred_mask,
            "image": img_result
        }

    def dummy_fastinpaint(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        image = inputs['image'].copy()
        mask = inputs['mask'].copy()
        kernel_size = 25
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        image_bgchanged = self.lamaInpaint.forward(image, None, mask, False)
        lama_res_np = image_bgchanged #[:,:,::-1]
        lama_res_PIL = Image.fromarray(image_bgchanged[:,:,::-1])
        gap = check_quality(lama_res_PIL, mask)
        if gap <= 30:
            denoise = 0.7
        elif 30 < gap <= 70:
            denoise = 0.8
        else:
            denoise = 0.9
        mask_ratio = np.sum(mask)/(np.sum(np.ones_like(mask))*255)
        print('mask_ratio', mask_ratio)
        if mask_ratio > 0.4:
            denoise = 1
        print('denoise', denoise)
        return {
            "image": lama_res_np,
            "score": denoise
        }


    def dummy_flux_inpaint(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
        unet_inpaint = loraloadermodelonly.load_lora_model_only(lora_name="inpaint.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_inpaint = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=unet_inpaint)[0]

        input_img_h, input_img_w = inputs["image"].shape[:2]
        input_img_PIL = inputs["image"].copy()
        input_img_PIL = input_img_PIL[:,:,::-1]
        input_img_PIL = Image.fromarray(input_img_PIL).convert('RGB') #tensor_to_pil(inputs["image"])
        image_ori_PIL = inputs["preimage"].copy()
        image_ori_PIL = image_ori_PIL[:,:,::-1]
        image_ori_PIL = Image.fromarray(image_ori_PIL).convert('RGB') #tensor_to_pil(inputs["preimage"])

        mask = inputs['mask'].copy()
        kernel_size = 25 #15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        input_mask_PIL = Image.fromarray(mask).convert('L') #mask_to_pil_l(inputs["mask"])
        image_lama = loadimage.load_image(image=image_ori_PIL)[0]
        inputMask = convert_mask_to_torch(np.array(input_mask_PIL))
        inputImage = convert_image_to_torch(np.array(input_img_PIL))
        denoise = inputs["score"]
        with torch.inference_mode():
            cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
            cliptextencode_pos = cliptextencode.encode(text="", clip=self.clip)[0]
            fluxguidance = self.FluxGuidance.append(guidance=30, conditioning=cliptextencode_pos)[0]
            latent_image = VAEEncode.encode(self.vae, image_lama)[0]
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            differentialdiffusion = Differentialdiffusion.apply(model=unet_inpaint)[0]
            inpaintmodelconditioning[2]['samples'] = latent_image['samples']
            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=denoise, #1.0,
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }
        

    def dummy_flux_fill(self, inputs: Dict[str, DataObject]) -> Dict[str, DataObject]:
     #   unet_fill = loraloadermodelonly.load_lora_model_only(lora_name="pytorch_lora_weights.safetensors", strength_model=1.0, model=self.unet_fill)[0]
        unet_fill = teacacheforimggen.apply_teacache(model_type="flux", rel_l1_thresh=0.4, model=self.unet_fill)[0]
        input_img_h, input_img_w = inputs["image"].shape[:2]
        input_img_PIL_np = inputs["image"].copy()
        input_img_PIL = input_img_PIL_np[:,:,::-1]
        input_img_PIL = input_img_PIL.astype(np.uint8)
        input_img_PIL = Image.fromarray(input_img_PIL).convert('RGB')
        input_mask_PIL = inputs["mask"].copy()
        input_mask_PIL = Image.fromarray(input_mask_PIL).convert('L') 
        
        with torch.inference_mode():
            class_flag = 0
            if inputs["preimage"] is not None:
                preimage = inputs["preimage"].copy()
                if preimage.shape[-1] == 3:
                    cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
                    cliptextencode_pos = cliptextencode.encode(text=inputs['prompt'], clip=self.clip)[0]
                    fluxguidance = self.FluxGuidance.append(guidance=30, conditioning=cliptextencode_pos)[0]
                    differentialdiffusion = Differentialdiffusion.apply(model=unet_fill)[0]

                 #   input_img_PIL, input_mask_PIL, original_size, max_ratios = prepare_expansion(input_img_PIL, input_mask_PIL)
                    input_img_PIL, input_mask_PIL, padding_info = pad_to_universal_symmetric(input_img_PIL, input_mask_PIL)
                    hh, ww = input_img_PIL.size[1], input_img_PIL.size[0]

                    size = 1024 
                    if ww > hh:
                        rate_wh = hh/ww
                        size_w = int(size)
                        size_h = int(rate_wh*size)
                    elif ww < hh:
                        rate_wh = ww/hh
                        size_h = int(size)
                        size_w = int(rate_wh*size)
                    else:
                        size_h = int(size)
                        size_w = int(size)
                    input_img_PIL = input_img_PIL.resize((size_w, size_h))
                    input_mask_PIL = input_mask_PIL.resize((size_w, size_h))

                    image_ori_PIL = inputs["preimage"].copy()
                    image_ori_PIL = image_ori_PIL[:,:,::-1]
                    image_ori_PIL = Image.fromarray(image_ori_PIL).convert('RGB') 
                    image_ori_PIL = image_ori_PIL.resize((size_w, size_h))
                    image_lama = loadimage.load_image(image=image_ori_PIL)[0]
                    latent_image = VAEEncode.encode(self.vae, image_lama)[0]
                    inputMask = convert_mask_to_torch(np.array(input_mask_PIL))
                    inputImage = convert_image_to_torch(np.array(input_img_PIL))
                    dd = random.uniform(0.9, 1.0) #0.94 #0.94 #0.99
                    class_flag = 1
                else:
                    cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
                  #  input_prompt = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but with " + inputs['prompt']
                    cliptextencode_pos = cliptextencode.encode(text=inputs['prompt'], clip=self.clip)[0]
                    fluxguidance = self.FluxGuidance.append(guidance=30, conditioning=cliptextencode_pos)[0]
                    differentialdiffusion = Differentialdiffusion.apply(model=unet_fill)[0]

                    image_ori_PIL = inputs["preimage"].copy()
                    image_ori_PIL = move_to_mask(image_ori_PIL, input_mask_PIL)
                    
                    image_ori_PIL[:, :, :3] = image_ori_PIL[:, :, 2::-1]
                    pil_preimage = Image.fromarray(image_ori_PIL, 'RGBA')
                    img = pil_preimage
                    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 注意这里是RGBA
                    image_left = Image.alpha_composite(white_bg, img)
                    image_left = image_left.convert('RGB')
                    

                    ww, hh = image_left.size[0], image_left.size[1]               
                    image_right = input_img_PIL
                    new_image = Image.new('RGB', (image_left.width + image_right.width, image_left.height))
                    new_image.paste(image_left, (0, 0))
                    new_image.paste(image_right, (image_left.width, 0))
                    combined_image = np.array(new_image)
                    width = ww
                    height = hh
                    black_mask = Image.new('L', (width, height), color=0)
                    combined_mask = Image.new('L', (width * 2, height))
                    combined_mask.paste(black_mask, (0, 0))
                    combined_mask.paste(input_mask_PIL, (width, 0))
                    mask_array = np.array(combined_mask)
                    inputMask = convert_mask_to_torch(mask_array)
                    inputImage = convert_image_to_torch(combined_image)
                    dd = 1.0
                    class_flag = 2
            else:
                cliptextencode_neg = cliptextencode.encode(text="", clip=self.clip)[0]
                cliptextencode_pos = cliptextencode.encode(text=inputs['prompt'], clip=self.clip)[0]
                fluxguidance = self.FluxGuidance.append(guidance=30, conditioning=cliptextencode_pos)[0]
                differentialdiffusion = Differentialdiffusion.apply(model=unet_fill)[0]

                inputMask = convert_mask_to_torch(np.array(input_mask_PIL))
                inputImage = convert_image_to_torch(np.array(input_img_PIL))
                dd = 1.0

            
            inpaintmodelconditioning = Inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=fluxguidance,
                negative=cliptextencode_neg,
                vae=self.vae,
                pixels=inputImage,
                mask=inputMask,
            )
            if class_flag == 1:
                inpaintmodelconditioning[2]['samples'] = latent_image['samples']

            ksampler = Ksampler.sample(
                seed=random.randint(1, 2**32),
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=dd, 
                model=differentialdiffusion,
                positive=inpaintmodelconditioning[0],
                negative=inpaintmodelconditioning[1],
                latent_image=inpaintmodelconditioning[2],
            )
            vaedecode = Vaedecode.decode(samples=ksampler[0], vae=self.vae)
            save_img = vaedecode[0]
            e_image = save_torch_tensor_as_image(save_img)
            if class_flag == 2:
                e_image = e_image.crop((ww,0,ww*2,hh))
            elif class_flag == 1:
                e_image = e_image.resize((ww, hh))
                e_image = crop_to_asymmetric(e_image, padding_info)
                    
            
        output_np = np.array(e_image)
        output_np = output_np[:,:,::-1]
        return {
            "image": output_np
        }

    
    def load_workflow(self, file_path: str) -> Dict[str, Any]:
        """加载JSON工作流文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except:
            workflow = file_path
        
        # 验证工作流基本结构
        if not all(key in workflow for key in ['process', 'pipeline']):
            raise ValueError("工作流JSON缺少必要字段(process或pipeline)")
        
        return workflow
    
    def parse_reference(self, ref: str) -> Tuple[Optional[str], str]:
        if not isinstance(ref, str):
            return (None, ref)
        
        # 处理 init[key] 格式
        if ref.startswith("init[") and ref.endswith("]"):
            return (None, ref[5:-1])
        
        # 处理 step...[...] 格式
        if ref.startswith("step") and "[" in ref and ref.endswith("]"):
            bracket_pos = ref.index("[")
            step_part = ref[4:bracket_pos]  # "step"后面的部分
            key_part = ref[bracket_pos+1:-1]
            
            # 如果step_part是数字（如"1"、"120"），则保留，否则视为None
            step = step_part if step_part.isdigit() else None
            return (step, key_part)
        
        return (None, ref)  # 不是引用格式


    def resolve_reference(self, ref: str, current_step: Optional[str] = None) -> Union[DataObject, list]:
        if not isinstance(ref, str):
            return ref
        if ',' in ref and ref.count('step') >= 1:
            refs = [r.strip() for r in ref.split(',')]
        else:
            refs = [ref.strip()]
        results = []
        for r in refs:
            if not r:
                continue
            if r.startswith(('init[', 'step')) and '[' in r and r.endswith(']'):
                results.append(self._resolve_single_reference(r, current_step))
            else:
                results.append(r)    
        return results[0] if len(results) == 1 else results


    def _resolve_single_reference(self, ref: str, current_step: Optional[str]) -> DataObject:
        step, key = self.parse_reference(ref)
        if step is None and key == ref:
            return ref
        if step is None:
            value = self.context.get(f"init_{key}")
            if value is None:
                raise ValueError(f"未找到初始输入: init_{key}")
            return value
        actual_step = step if step is not None else current_step
        if actual_step is None:
            raise ValueError(f"无法确定步骤编号: {ref}")
        print(222222222222222222)
        print(actual_step, key)
        value = self.context.get(f"step{actual_step}_{key}")
        if value is None:
            raise ValueError(f"未找到步骤输出: step{actual_step}_{key}")
        return value

    def validate_data_type(self, key: str, value: DataObject):
        print('not valid')
    
    def execute_step(self, step_def: Dict[str, Any]):
        """执行单个工作流步骤"""
        step_num = str(step_def.get("step"))  # 确保step_num是字符串
        model_name = step_def["model"]
        
        print(f"\n=== 执行步骤 {step_num} ({model_name}) ===")
        
        # 准备输入
        inputs = {}
        for input_key, input_ref in step_def["input"].items():
            try:
                # 解析输入引用
                resolved_value = self.resolve_reference(input_ref, current_step=step_num)
                # 特殊处理ratio转换
                if input_key == "ratio" and isinstance(resolved_value, str):
                    try:
                        resolved_value = float(resolved_value)
                    except ValueError as e:
                        raise ValueError(f"无法将ratio转换为数值: {resolved_value}") from e
                
                # 验证数据类型
                self.validate_data_type(input_key, resolved_value)
                inputs[input_key] = resolved_value
            except Exception as e:
                raise ValueError(f"步骤 {step_num} 输入 {input_key} 解析失败: {e}") from e
        
        # 获取模型函数
        model_func = self.model_mapping.get(model_name)
        if model_func is None:
            raise ValueError(f"未知模型: {model_name}")
        
        outputs = model_func(inputs)
        
        # 存储输出
        for output_key, output_ref in step_def["output"].items():
            try:
                # 解析输出引用目标
                ref_step, ref_key = self.parse_reference(output_ref)
                actual_step = ref_step if ref_step is not None else step_num
                
                if actual_step is None:
                    raise ValueError(f"无法确定输出存储位置: {output_ref}")
                
                # 验证模型返回了所需的输出
                if output_key not in outputs:
                    raise ValueError(f"模型 {model_name} 没有返回预期的输出 {output_key}")
                
                # 验证输出数据类型
                self.validate_data_type(ref_key, outputs[output_key])
                
                # 存储到上下文
                storage_key = f"step{actual_step}_{ref_key}"
                self.context[storage_key] = outputs[output_key]
                print(f"存储输出: {storage_key} = {type(outputs[output_key])}", 
                      f"形状: {outputs[output_key].shape}" if hasattr(outputs[output_key], 'shape') else "")
                
            except Exception as e:
                raise ValueError(f"步骤 {step_num} 输出 {output_key} 处理失败: {e}") from e
    
    def execute(self, workflow_file, initial_inputs: Optional[Dict[str, DataObject]] = None):
        """执行整个工作流"""
        self.workflow = self.load_workflow(workflow_file)
        print(self.workflow)
        self.context = {}
        print("="*50)
        print(f"开始执行工作流: {self.workflow['process']}")
        print("="*50)
        
        # 初始化上下文
        if initial_inputs:
            for key, value in initial_inputs.items():
                # 处理图像路径
                print('0000000000000000',key, value)
                if key == "image" and isinstance(value, dict):#str):
                    img = value['background'].convert('RGB')
                    img = np.array(img)[:,:,::-1]
                    size = 1024
                    ww, hh = img.shape[1], img.shape[0]
                    if ww%64 == 0 and hh%64 == 0 and max(ww,hh) <= 1024:
                        size_w = ww
                        size_h = hh
                    else:
                        if ww > hh:
                            rate_wh = hh/ww
                            size_w = int(size)
                            size_h = int(((rate_wh*size) + 64) // 64 * 64)
                        elif ww < hh:
                            rate_wh = ww/hh
                            size_h = int(size)
                            size_w = int(((rate_wh*size) + 64) // 64 * 64)
                        else:
                            size_h = int(size)
                            size_w = int(size)
                    img = cv2.resize(img, (size_w, size_h))
                    if img is None:
                        raise ValueError(f"无法加载图像: {value}")
                    self.context[f"init_{key}"] = img
                    print(f"加载初始图像: {value} → 形状: {img.shape}")
                elif key == "mask" and isinstance(value, dict):
                    mask = value['layers'][0]
                    mask = np.array(mask)
                    mask = mask[:, :, -1]
                    size = 1024
                    ww, hh = mask.shape[1], mask.shape[0]
                    if ww%64 == 0 and hh%64 == 0 and max(ww,hh) <= 1024:
                        size_w = ww
                        size_h = hh
                    else:
                        if ww > hh:
                            rate_wh = hh/ww
                            size_w = int(size)
                            size_h = int(((rate_wh*size) + 64) // 64 * 64)
                        elif ww < hh:
                            rate_wh = ww/hh
                            size_h = int(size)
                            size_w = int(((rate_wh*size) + 64) // 64 * 64)
                        else:
                            size_h = int(size)
                            size_w = int(size)
                    mask = cv2.resize(mask, (size_w, size_h))
                    self.context[f"init_{key}"] = mask
                else:
                    self.validate_data_type(key, value)
                    self.context[f"init_{key}"] = value
                    print(f"初始输入 {key}: {type(value)}")
        
        # 执行每个步骤
        try:
            for step in self.workflow["pipeline"]:
                if "step" in step:
                    self.execute_step(step)
                elif "result" in step:
                    result_ref = step["result"]
                    if isinstance(result_ref, list):
                        result_ref = "[" + ", ".join(result_ref) + "]"
                    result_ref = result_ref.replace("'", "")

                    if result_ref.startswith("[") and result_ref.endswith("]"):
                        result_ref = result_ref[1:-1]

                    json_str = str(copy.deepcopy(workflow_file)  )
                    json_str = json_str.replace('"result": "[' + result_ref + ']"','')
                    result_ref = re.sub(r'step\d+\[mask\],?\s*', '', result_ref)
                    if result_ref.count('step') == 1:
                        result_ref = result_ref.replace(',','')
                    result_ref = filter_a_str(result_ref, json_str)
                    print('result_ref', result_ref)
                    result = self.resolve_reference(result_ref)
                    print("\n" + "="*50)
                    print("工作流执行完成！最终结果:")
                    if isinstance(result, list):
                        output_p = "workflow_result"
                        for ss in range(len(result)):
                            if ss < 10:
                                output_path = output_p + str(ss) + '.jpg'
                                cv2.imwrite(output_path, result[ss])
                                print(f"结果已保存为: {output_path}")
                    elif isinstance(result, np.ndarray):
                        print(f"图像结果，形状: {result.shape}")
                        output_path = "workflow_result.jpg"
                        cv2.imwrite(output_path, result)
                        print(f"结果已保存为: {output_path}")
                    else:
                        print(f"文本结果: {result}")
                    
                    return result
        except Exception as e:
            print("\n" + "!"*50)
            print(f"工作流执行失败: {e}")
            print("当前上下文键:", self.context.keys())
            raise
        
        raise RuntimeError("工作流没有定义结果输出")
    

import gradio as gr
from argparse import ArgumentParser

class Demo():
    def __init__(self, args):
        self.args = args

        qwen_workflow_dir = "./Builder"

        self.llm = LLM(
            model= qwen_workflow_dir,
            limit_mm_per_prompt={'image': 1},
            gpu_memory_utilization=0.3, ##0.35, #0.58,
            dtype='float16',
            enable_lora=True,
            max_lora_rank=32
        )
        self.sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=4096,stop_token_ids=[])

        self.processor_workflow = AutoProcessor.from_pretrained(qwen_workflow_dir)

        self.executor = WorkflowExecutor(llm=self.llm, sampling_params=self.sampling_params, processor_workflow=self.processor_workflow)

        self.demo_init()
        
    def demo_init(self):
        with gr.Blocks(title="Lego-Edit", css="style.css", fill_width=True, fill_height=True) as demo:
            state = gr.State({
                
            })
            with gr.Row():
                gr.Markdown("<div align='center'><font size='18'>Lego-Edit</font></div>")
            gr.HTML("""
                    <div style="text-align: right;">
                    </div>
                    """)
            
            with gr.Row(variant='panel'):
                with gr.Column(scale=2):
                    gr.Markdown("### Input region")
                    # 使用ImageMask实现涂抹生成掩码功能
                    input_image = gr.ImageMask(
                        label="Click & Drag to create mask",
                        type='pil',
                        width="100%",
                        height="66vh",
                        interactive=True,
                        show_download_button=True
                    )
                    prompt = gr.Textbox(label="input prompt")
                    # input_image = gr.ImageEditor(label="", visible=True, eraser=False, layers=False)
                    with gr.Row():
                        run_button = gr.Button(value="运行", visible=True)
                        clear_button = gr.Button(value="清空", visible=True)
                with gr.Column(scale=3):
                    gr.Markdown("### Output region")
                    # output_image = gr.Image(label="")
                    output_image = gr.Gallery(label="output image", columns=3)
                    save_file = gr.File(label="save image", visible=False)
                    output_cot = gr.TextArea(label="output CoT")
                    data = gr.State()
                    
            # function
            run_button.click(
                self.get_cot, [input_image, prompt], [output_cot, data]
            ).then(
                self.inference, [input_image, data], [output_image, save_file]
            )
            
        self.demo = demo

    def get_cot(self, input_image, prompt):
        mask1 = input_image['layers'][0]
        mask1 = np.array(mask1)
        mask1 = mask1[:, :, -1]
        if np.sum(mask1) != 0:
            prompt = '用户提供了mask并希望' + prompt

        with open('prompt_eng_0812_7b.txt', 'r', encoding='utf-8') as file:
            ppp = file.read()
        ppp = ppp.replace('YourInstruction', prompt)

        iii = input_image['background'].convert('RGB') #Image.open(input_image)

        size = 448
        ww, hh = iii.size[0], iii.size[1]
        if ww > hh:
            rate_wh = hh/ww
            size_w = int(size)
            size_h = int(rate_wh*size)
        elif ww < hh:
            rate_wh = ww/hh
            size_h = int(size)
            size_w = int(rate_wh*size)
        else:
            size_h = int(size)
            size_w = int(size)
        iii = iii.resize((size_w, size_h))

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": iii, 
                    },
                    {"type": "text", "text": ppp},
                ],
            }
        ]
        ''' 
        text = self.processor_workflow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs_qwen = self.processor_workflow(
                text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs_qwen = inputs_qwen.to("cuda")
        generated_ids = self.qwen_workflow.generate(**inputs_qwen, max_new_tokens=4096)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_qwen.input_ids, generated_ids)]
        output_text = self.processor_workflow.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = output_text[0]
        '''
        prompt_post = self.processor_workflow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        llm_inputs = {
            'prompt': prompt_post,
            'multi_modal_data': mm_data,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        answer = outputs[0].outputs[0].text
        
        think_pattern = r'<think>(.*?)</think>'
        text1 = re.search(think_pattern, answer, re.DOTALL).group(1).strip()
        answer_pattern = r'<answer>(.*?)</answer>'
        text2 = answer.split("</think>")[-1] #re.search(answer_pattern, answer, re.DOTALL).group(1).strip()
        data = json5.loads(text2)
        data = modify_json_mask(data)
        data = fix_json_structure(data)
        with open('workflow.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return text1, data

    def inference(self, input_image, data):
        gradio_rgb = []
        tmp_files = []
        try:
            initworkflow = copy.deepcopy(data) #self.path_json
            initworkflow = initworkflow["pipeline"][0]["input"]
            if initworkflow["image"] == "init[image]":
                initworkflow["image"] = input_image
                initworkflow["mask"] = input_image
            initial_inputs = initworkflow 

            # 执行工作流
            result = self.executor.execute(data, initial_inputs)

            if isinstance(result, list):
                for img in result:
                    img = process_image(img)
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    image = Image.fromarray(img[:, :, ::-1])
                    image.save(tmp.name)
                    tmp_files.append(tmp.name)
                    gradio_rgb.append(img[:, :, ::-1])
            elif isinstance(result, np.ndarray):
                result = process_image(result)
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                image = Image.fromarray(result[:, :, ::-1])
                image.save(tmp.name)
                tmp_files.append(tmp.name)
                gradio_rgb.append(result[:, :, ::-1])
        except:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image = Image.open(input_image) 
            image.save(tmp.name)
            tmp_files.append(tmp.name)
            gradio_rgb.append(input_image)
        return gradio_rgb, gr.update(value=tmp_files, visible=True)
        
    def launch(self):
        self.demo.queue()
        self.demo.launch(share=False, 
                         server_name='xx.xx.xx.xx',
                         server_port=xx, #8000
                         show_api=False, 
                         show_error=True, 
                         max_threads=1)
        
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save-path', type=str, default='./saved_inputs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    demo = Demo(args)
    demo.launch()
