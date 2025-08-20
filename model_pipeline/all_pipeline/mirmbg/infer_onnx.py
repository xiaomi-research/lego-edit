import torch
import numpy as np
import os
from PIL import Image
import onnxruntime as ort

import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from datetime import datetime

class segment_MIRMBG:
    def __init__(self,
                rootpath,
                model_input_size=[1024, 1024],
                ):
        T0 = datetime.now()
        onnx_model_path = os.path.join(rootpath, '20240410_mibg.onnx')
        # onnx_model_path = 'model.onnx'
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess = ort.InferenceSession(onnx_model_path, so, providers=['CUDAExecutionProvider'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_size = model_input_size
        self.saving = False
        if self.saving:
            self.save_root_path = './saving_dir'
            if not os.path.exists(self.save_root_path):
                os.makedirs(self.save_root_path)
        T1 = datetime.now()
        latency1 = (T1-T0).total_seconds()*1000.0
        latency1 = round(latency1, 3)
        print(f"segment init consume: {latency1} ms")

    def forward(self, orig_im):
        T0 = datetime.now()
        orig_im_size = orig_im.shape[0:2]        
        image = self.preprocess_image(orig_im, self.model_input_size).to(self.device)
        T1 = datetime.now()
        latency1 = (T1-T0).total_seconds()*1000.0
        latency1 = round(latency1, 3)
        print(f"segment preprocess time consume: {latency1} ms")
        output = self.sess.run(None, {"input":image.cpu().detach().numpy()})
        T2 = datetime.now()
        latency1 = (T2-T1).total_seconds()*1000.0
        latency1 = round(latency1, 3)
        print(f"segment infer time consume: {latency1} ms")
        # print(output[0].shape)
        
        result_image = self.postprocess_image(torch.from_numpy(output[0]), orig_im_size)

        return result_image

    def forward_imgpath(self, im_path, ):
        orig_im = Image.open(im_path).convert("RGB")
        orig_im = np.array(orig_im)
        orig_im_size = orig_im.shape[0:2]
        if orig_im.shape[2]>3:
            orig_im = orig_im[:,:,:3]
        
        image = self.preprocess_image(orig_im, self.model_input_size).to(self.device)

        output = self.sess.run(None, {"input":image.cpu().detach().numpy()})
        # print(output[0].shape)
        
        result_image = self.postprocess_image(torch.from_numpy(output[0]), orig_im_size)

        if self.saving:
            pil_im = Image.fromarray(result_image)
            no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
            orig_image = Image.open(im_path)
            no_bg_image.paste(orig_image, mask=pil_im)
            no_bg_image.save(os.path.join(self.save_root_path, os.path.splitext(im_path.split('/')[-1])[0]+'.png'))
        else:
            # print(result_image.shape)
            return result_image

    def preprocess_image(self, im, model_input_size):
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear').type(torch.uint8)
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
        return image

    def postprocess_image(self, result: torch.Tensor, im_size: list)-> np.ndarray:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array


if __name__=='__main__':
    rmbg  = segment_MIRMBG('model.onnx')
    T0 = datetime.now()
    rmbg.forward_imgpath('rebg_1.png')
    T1 = datetime.now()
    latency1 = (T1-T0).total_seconds()*1000.0
    latency1 = round(latency1, 3)
    print(f"subjectSeg time consume: {latency1} ms")

    rmbg.forward_imgpath('rebg_1.png')
    T2 = datetime.now()
    latency1 = (T2-T1).total_seconds()*1000.0
    latency1 = round(latency1, 3)
    print(f"subjectSeg time consume: {latency1} ms")

    import cv2
    img = cv2.imread('rebg_1.png')
    rmbg.forward(img)
    T3 = datetime.now()
    latency1 = (T3-T2).total_seconds()*1000.0
    latency1 = round(latency1, 3)
    print(f"subjectSeg time consume: {latency1} ms")