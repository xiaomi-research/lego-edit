from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from .efficientvit import efficientvit_sam_xl0
from .efficientvit.utils import load_state_dict_from_file
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig
import numpy as np
import copy

def apply_coords(coords, original_size, target_size):
    old_h, old_w = original_size
    
    if old_w >= old_h:
        scale = target_size / old_w
    else:
        scale = target_size / old_h
    
    coords = copy.deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * scale
    coords[..., 1] = coords[..., 1] * scale
    return coords.reshape(-1, 4)


def get_bbox(mask, pad):
    
    inds = np.where(mask > 0)
    

    x_min_bound = 0
    y_min_bound = 0
    x_max_bound = mask.shape[1] - 1
    y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return [x_min, y_min, x_max - x_min, y_max - y_min]

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        print('initializing linear layer...')
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class EvfSamEffiVitModel(PreTrainedModel):
    config_class = EvfConfig
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(EvfSamEffiVitModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", None)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)

    def init_weights_cyj(self):
        print("before: ", self.text_hidden_fcs_cyj[0].state_dict()['0.weight'])
        self.text_hidden_fcs_cyj.apply(init_weights)
        print("after: ", self.text_hidden_fcs_cyj[0].state_dict()['0.weight'])

    def check_encoder_params(self):
        weight = load_state_dict_from_file(self.vision_pretrained)
        model_state = self.visual_model.state_dict()
        for k,v in weight.items():
            if not v.equal(model_state[k]):
                print("XXXXXXXXXXXXXXXXXXXXXx not match : ", k)
            else:
                print("!!!!! match", k)
    
    def load_vision_model(self):
        weight = load_state_dict_from_file(self.vision_pretrained)
        self.visual_model.load_state_dict(weight)

    def initialize_evf_modules(self, config):
        # SAM
        if config.sam_scale=="huge":
            self.visual_model = efficientvit_sam_xl0()
          #  weight = load_state_dict_from_file(self.vision_pretrained)
          #  self.visual_model.load_state_dict(weight)
        else:
            raise NotImplementedError
        
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        if self.train_prompt_encoder:
            self.visual_model.prompt_encoder.no_mask_embed.requires_grad_(True)
            
        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"model config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(
                beit_state_dict, 
                strict=False
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True
                
        # Projection layer
        in_dim = config.hidden_size
        assert in_dim==beit_config.encoder_embed_dim, \
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        text_fc_cyj = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 2*out_dim),
            nn.ReLU(),
            nn.Linear(2*out_dim, 2*out_dim),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self.text_hidden_fcs_cyj = nn.ModuleList([nn.Sequential(*text_fc_cyj)])
        self.text_hidden_fcs_cyj.train()
        for param in self.text_hidden_fcs_cyj.parameters():
            param.requires_grad = True
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(
        self,
        images: torch.FloatTensor,
        images_evf: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        multimask_output = False
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf, 
            textual_tokens=input_ids, 
            text_padding_position=~attention_masks
            )

        feat = output["encoder_out"][:, :1, ...]

        feat_ori = self.text_hidden_fcs[0](feat)
        feat_ori = torch.split(feat_ori, [offset[i+1] - offset[i] for i in range(len(offset)-1)])
        
        feat = self.text_hidden_fcs_cyj[0](feat)
        feat = feat.reshape(-1, 2, 256)
        feat = torch.split(feat, [offset[i+1] - offset[i] for i in range(len(offset)-1)])

        pred_masks = []
        pred_embedding_list = []
        target_embedding_list = []
        for i in range(len(feat)):
            mask_cur = masks_list[i]
            
            curboxes = []
            
            for j in range(len(mask_cur)):
            
                bboxs = get_bbox(mask_cur[j].cpu().detach().numpy(), pad=0)
                bboxs = np.array(bboxs)
                bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2] # x2 = x1 + w
                bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3] # y2 = y1 + h
                bboxs = apply_coords(bboxs.reshape(-1,2,2), mask_cur.shape[1:], 1024)
                curbox = torch.tensor(bboxs, dtype=torch.float32, device=feat[i].device)
                curboxes.append(curbox)
            
            curboxes = torch.stack(curboxes, dim=0)
            (
                sparse_embeddings_box,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=curboxes,
                masks=None,
                text_embeds=None,
            )
            sparse_embeddings_box = sparse_embeddings_box.to(feat[i].dtype)
            ## calulate loss
            pred_embedding = feat[i]
            target_embedding = sparse_embeddings_box
            pred_embedding_list.append(pred_embedding)
            target_embedding_list.append(target_embedding)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=torch.cat([feat[i], feat_ori[i]], dim=1),#feat[i],
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            if 0:
                import cv2
                pred_mask = self.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=label_list[i].shape,
                )
                mask_sav = pred_mask[0][0].detach().cpu().numpy()
                # mask_sav1 = pred_mask[1][0].detach().cpu().numpy()
                cv2.imwrite('./cccccccccccccccccyj_efficient_box_sup_featori+feat.png', 255*mask_sav.astype(np.uint8))
                # cv2.imwrite('./cccccccccccccccccyj_1.png', 255*mask_sav1.astype(np.uint8))

            if multimask_output:
                sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
                low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]
          
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        mask_bce_loss = 0
        mask_dice_loss = 0
        loss_embedding = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            loss_embedding += (nn.functional.mse_loss(
                                            pred_embedding_list[batch_idx],
                                            target_embedding_list[batch_idx],
                                            reduction='mean'
                                        ) * gt_mask.shape[0]
                                )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_embedding_loss = 1 * loss_embedding / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss + mask_embedding_loss

        loss = mask_loss

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_embedding_loss": mask_embedding_loss,
            "mask_loss": mask_loss,
        }
    
    def inference(
            self,
            images,
            images_evf,
            input_ids,
            resize_list,
            original_size_list,
            multimask_output=False,
        ):
        with torch.no_grad():
            image_embeddings = self.visual_model.image_encoder(images)
        multimask_output = multimask_output

        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids, text_padding_position=torch.zeros_like(input_ids))

        feat = output["encoder_out"][:, :1, ...]
        
        feat_ori = self.text_hidden_fcs[0](feat)
        
        feat = self.text_hidden_fcs_cyj[0](feat)
        feat = feat.reshape(-1, 2, 256)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=torch.cat([feat, feat_ori], dim=1),
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        low_res_masks, iou_predictions = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=torch.cat([feat, feat_ori], dim=1),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        if multimask_output:
            sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
            low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

        pred_mask = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=resize_list[0],
            original_size=original_size_list[0],
        )

        return pred_mask[:, 0]


AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, EvfSamEffiVitModel)
