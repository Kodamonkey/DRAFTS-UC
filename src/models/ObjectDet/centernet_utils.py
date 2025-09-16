# This module provides utility functions for CenterNet inference.

import numpy as np
import torch, cv2
from torch import nn
from torchvision.ops import nms


# This function denormalizes.
def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    img *= std
    img += mean
    img  = (img * 255).astype(np.uint8)

    return img


# This function pools nms.
def pool_nms(heat, kernel=3):

    pad  = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


# This function decodes bbox.
def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence, cuda):
                                                                               
                                    
                                    
                                      
                                       
                         
                                                                               
    pred_hms = pool_nms(pred_hms)
    b, c, output_h, output_w = pred_hms.shape
    detects = []
                                                                               
                       
                                                                               
    for batch in range(b):
                                                                                   
                                                       
                                                            
                                                                                        
                                                               
                                                                                   
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        xv, yv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w), indexing='xy')
                                                                                   
                                                
                                                
                                                                                   
        xv, yv      = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()
                                                                                   
                                                 
                                              
                                                                                   
        class_conf, class_pred  = torch.max(heat_map, dim=-1)
        mask                    = class_conf > confidence

                                                   
                        
                                                   
        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

                                                  
                       
                                                  
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
                                                  
                    
                                                  
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
                                                  
                         
                                                  
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects


# This function postprocesses.
def postprocess(prediction, need_nms, input_shape, nms_iou=0.4):
    output = [None for _ in range(len(prediction))]

                                                                
                       
                                                                
    for i, image_pred in enumerate(prediction):
        detections        = prediction[i]
        if len(detections) == 0:
            continue
                                                    
                          
                                                    
        unique_labels     = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections    = detections.cuda()

        for c in unique_labels:
                                                        
                                 
                                                        
            detections_class    = detections[detections[:, -1] == c]
            if need_nms:
                                                            
                                        
                                                            
                keep            = nms(detections_class[:, :4], detections_class[:, 4], nms_iou)
                max_detections  = detections_class[keep]
            else:
                max_detections  = detections_class

            output[i]           = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            output[i][:, :4]   *= input_shape

    return output


# This function extracts detection results from model outputs.
def get_res(hm, wh, offset, confidence):

    outputs = decode_bbox(hm, wh, offset, confidence, cuda=True)
    results = postprocess(outputs, True, 512, nms_iou=0.3)

    if results[0] is None:
        return None, None

    top_conf    = results[0][:, 4]
    top_boxes   = results[0][:, :4]

    return top_conf, top_boxes
