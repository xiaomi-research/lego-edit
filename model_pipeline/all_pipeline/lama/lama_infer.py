import logging
import os
import sys
import traceback
from PIL import Image
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
sys.path.append(os.path.join(os.getcwd(), "lama"))
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "lama")

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
current_dir_parent = os.path.dirname(current_dir)
# print(current_path)
# print(current_dir)
# print(current_dir_parent)


sys.path.append(os.path.join(current_dir, "lama"))
os.environ['TORCH_HOME'] = os.path.join(current_dir, "lama")
from saicinpainting.training.trainers import load_checkpoint

from scipy import ndimage


class lama_Inpaint:
    def __init__(self, ):
        device = 'cuda:0' #torch.device(predict_config.device)
        train_config_path = os.path.join('./lama/big-lama/config.yaml') #TODO: lama-fourier
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'
            checkpoint_path = os.path.join('./lama/big-lama/models/best.ckpt') #TODO: lama-fourier
            self.model_lama = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
            self.model_lama.to('cuda')
            self.model_lama.freeze()
    
    def compute_mask(self, srcout, errorout): # srcout是原始图像的Mask, errorout是sd生成的图像的Mask
        flag = 0
        num_labels_1, labels_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(srcout, connectivity=8)
        num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(errorout, connectivity=8)
        selected_labels_1 = []
        selected_areas_1 = []
        all_area1 = srcout.sum()/255
        for i in range(1, num_labels_1):
            area = stats_1[i, cv2.CC_STAT_AREA]
            cur_ratio = area/all_area1
            if cur_ratio > 0.1:
                selected_labels_1.append(i)
                selected_areas_1.append(cur_ratio)
        selected_labels_2 = []
        selected_areas_2 = []
        all_area2 = errorout.sum()/255
        for i in range(1, num_labels_2):
            area = stats_2[i, cv2.CC_STAT_AREA]
            cur_ratio = area/all_area1
            if cur_ratio > 0.1:
                selected_labels_2.append(i)
                selected_areas_2.append(cur_ratio)

        all_diff = np.zeros_like(srcout)
        new_masks = np.zeros_like(srcout)

        for i in selected_labels_1:
            cur_src_mask = labels_1==i
            for j in selected_labels_2:
                cur_hbj_mask = labels_2==j
                intersection = np.logical_and(cur_src_mask, cur_hbj_mask)
                iou_1 = np.sum(intersection) / np.sum(cur_src_mask)
                iou_2 = np.sum(intersection) / np.sum(cur_hbj_mask)

                if iou_1 > 0.85 and iou_2 > 0.85:
                    diff = (cur_hbj_mask.astype(np.float32) - cur_src_mask.astype(np.float32))>0
                    all_diff = all_diff+diff
                    new_mask = cur_hbj_mask.astype(np.float32) 
                    new_masks = new_masks+new_mask

        diff = 255*all_diff.astype(np.uint8)
        masks = 255*new_masks.astype(np.uint8)
        ratio = abs(diff).sum()/(255*diff.size)
        if ratio > 0.05:
            flag = 1
        return masks, flag

    

    def forward(self, img_sd, mask_ori, mask_sd, com_flag):
        if com_flag:
            tmp_height, tmp_width = mask_sd.shape
            mask_ori = cv2.resize(mask_ori, (tmp_width, tmp_height))
            mask_ori = np.where(mask_ori>127, 255 ,0).astype(np.uint8)
            mask_sd = np.where(mask_sd>127, 255 ,0).astype(np.uint8)
            mask_new, flag = self.compute_mask(mask_ori, mask_sd)
        else:
            flag = 0
        if flag == 0:
            mask_new = mask_sd
            maskq = mask_new.astype(np.float32) / 255.0 #mask_onestage.astype(np.float32) / 255.0
            maskq = maskq[None, None]
            maskq[maskq < 0.5] = 0
            maskq[maskq >= 0.5] = 1
            maskq = np.squeeze(maskq)
            kernel = np.ones((20,20), np.uint8)
            maskq = ndimage.binary_dilation(maskq, structure=kernel).astype(maskq.dtype)
            maskq = np.expand_dims(np.expand_dims(maskq, axis=0), axis=0)
            maskq = torch.from_numpy(maskq)
            i_torch = torch.from_numpy(np.array(img_sd)[None].transpose(0, 3, 1, 2)).to(dtype=torch.float32)
            batch = {}
            batch['image'] = (i_torch/255).cuda()
            batch['mask'] = maskq.cuda()
            batch = self.model_lama(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8') 
            return cur_res
        else:
            return img_sd
        


if __name__=='__main__':
    img = np.array(Image.open(os.path.join(current_dir, 'rebg_4.png')).convert("RGB"))
    mask_ori = np.array(Image.open(os.path.join(current_dir, 'srcout.png')).convert("L"))
    mask_sd = np.array(Image.open(os.path.join(current_dir, 'errorout.png')).convert("L"))

    model = lama_Inpaint()
    img_filled = model.forward(img, mask_ori, mask_sd)
    cv2.imwrite(os.path.join(current_dir, 'savedres.png'), img_filled)



