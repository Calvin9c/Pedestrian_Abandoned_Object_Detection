# TorchReID
from __future__ import print_function, division


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import os
import sys
# TorchReID
sys.path.append("deep_person_reid")
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative

import time

# Usr Config
import argparse
from usr_config import tracking_config

import functions as our_funcs

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Process, Queue, Lock

from natsort import natsorted
import json

import numpy as np
import math
import cv2

import platform

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
import torchvision.utils as utils
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, our_prcoess_bin_masks
from utils.segment.plots import plot_masks, our_plot_masks
from utils.torch_utils import select_device, smart_inference_mode


def cos_sim(a, b):
    a = np.mat(a)
    b = np.mat(b)
    return float(a * b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

# 以下是演算法組寫的（以上是reID model相關）
def check_sizediff(box1, box2): 
    if box1[2]/box2[2] > 2 or box2[2]/box1[2] > 2 or box1[3]/box2[3] > 2 or box2[3]/box1[3] > 2:
        return True

from torchvision import transforms
def get_similarity(extractor, box1, box1_rgb, box2, box2_rgb):
    height, width, _ = box1_rgb.shape
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ul, lr = our_funcs.dilate_bbox(box1, height, width)
    y_cropped_img = box1_rgb[ul[1]:lr[1], ul[0]:lr[0]]

    ul, lr = our_funcs.dilate_bbox(box2, height, width)
    x_cropped_img = box2_rgb[ul[1]:lr[1], ul[0]:lr[0]]

    img1 = y_cropped_img
    img2 = x_cropped_img

    image_list = [
        img1,img2
    ]

    features = extractor(image_list)
    features = features.cpu()

    sim_score = cos_sim(features[0],features[1])

    return sim_score

def get_moved_dist(x_box, y_box):
    y_midpoint_x = y_box[0] + y_box[2]/2
    y_midpoint_y = y_box[1] + y_box[3]/2
    x_midpoint_x = x_box[0] + x_box[2]/2
    x_midpoint_y = x_box[1] + x_box[3]/2
    dist = math.sqrt((x_midpoint_x - y_midpoint_x)**2 + (x_midpoint_y - y_midpoint_y)**2)
    return dist

from scipy.optimize import linear_sum_assignment
def hungarian_algorithm(cost_matrix):
    n, m = cost_matrix.shape
    matches = np.full(n, -1, dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i in range(len(row_ind)):
        matches[row_ind[i]] = col_ind[i]
    return matches

def check_covered(z_box, x_box):
    overlap_left_x = max(z_box[0], x_box[0])
    overlap_left_y = max(z_box[1], x_box[1])
    overlap_right_x = min(z_box[0] + z_box[2], x_box[0] + x_box[2])
    overlap_right_y = min(z_box[1] + z_box[3], x_box[1] + x_box[3])

    overlap_area = (overlap_right_x - overlap_left_x) * (overlap_right_y - overlap_left_y) if overlap_left_x < overlap_right_x and overlap_left_y < overlap_right_y else 0

    return overlap_area / (z_box[2] * z_box[3])

def check_yolo_covered(z_box, x_box):
    overlap_left_x = max(z_box[0], x_box[0])
    overlap_left_y = max(z_box[1], x_box[1])
    overlap_right_x = min(z_box[0] + z_box[2], x_box[0] + x_box[2])
    overlap_right_y = min(z_box[1] + z_box[3], x_box[1] + x_box[3])

    overlap_area = (overlap_right_x - overlap_left_x) * (overlap_right_y - overlap_left_y) if overlap_left_x < overlap_right_x and overlap_left_y < overlap_right_y else 0

    min_area = min((z_box[2] * z_box[3]), (x_box[2] * x_box[3]))

    return overlap_area / min_area

def check_in_yolo(yolo_box, x_box): # return True means detected by yolo --> not garbage / False --> maybe garbage
    for box in yolo_box:
        z_box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        if check_yolo_covered(z_box, x_box) > 0.8:
            return max((z_box[2]*z_box[3]), (x_box[2]*x_box[3])) / min((z_box[2]*z_box[3]), (x_box[2]*x_box[3])) 
    return 0

def check_coverby_delcc(delcc_box, x_box):
    for box in delcc_box:
        if check_covered(x_box, box) > 0.9 :
            return True
    return False
    
def check_in_coverbyyolo(cover_by_yolo, x_box):
    for box in cover_by_yolo:
        z_box = box["bbox"]
        if check_covered(x_box, z_box) > 0.9 :
            return True
    return False

def add_trackinglist(wait_to_add, y_box, current_frame_idx):
    tmp = {
        "bbox": y_box,
        "frame_idx": current_frame_idx,
        "ref": True,
        "duration": 1,
        "state": 0,
        "no_pair_counter": 0, 
        "is_garbage": False,
        "moved_dist" : 0,
        "check_yolo_cnt" : 0,  # number of times that has checked in yolo, if == 3 --> check not_in_yolo
        "not_in_yolo" : 0,
        "be_covered" : 0, 
        "appear_bbox" : y_box
    }
    wait_to_add.append(tmp)
    return wait_to_add

def update_trackinglist(trackinglist_info, y_box):
    x_box = trackinglist_info["bbox"]
    trackinglist_info["moved_dist"] += get_moved_dist(x_box, y_box)
    trackinglist_info["bbox"] = y_box
    trackinglist_info["ref"] = True
    trackinglist_info["duration"] += 1
    trackinglist_info["state"] = 1
    trackinglist_info["no_pair_counter"] = 0
    trackinglist_info["be_covered"] = 0

    return trackinglist_info



@smart_inference_mode()
def run(
        # For Multiprocessing
        bgs_frames = None,
        rgb_frames = None,
        cc_bboxes_info = None,
        cc_del_info = None,
        lock_for_dict = None,
        run_yolo_only = False,

        # weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        # weights=ROOT / 'yolov7s-seg.pt',  # model.pt path(s)
        weights='yolov7-seg.pt', # model.pt path(s)

        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)(544, 960)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 64],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
):
   
    # usr config
    debug_mode = False

    # Threshold    
    # 23/10/03以前底下th設定分別為115, 0.15, 495
    light_wh_area_th = 200 # lightweight_denoise 180
    fg_wh_pixels_ratio_th = 0.15 # reinforce fg
    wh_area_th = 900 # denoise

    # video_name = source.split('/')[1] # 原本
    
    video_name = source.split('/')[-1]
    video_name = video_name.split('.')[0]
    print(f"video_name: {video_name}")

    # 是否只需要跑yolo?
    if run_yolo_only:
        # 1.
        assert rgb_frames == None
        # 2.
        assert cc_bboxes_info == None
        save_pth_for_res_cc = f"./{video_name}/cc_res_npys"
        if not os.path.exists(save_pth_for_res_cc): os.makedirs(save_pth_for_res_cc)
        # 3.
        assert cc_del_info == None
        save_pth_for_del_cc = f"./{video_name}/cc_del_npys"
        if not os.path.exists(save_pth_for_del_cc): os.makedirs(save_pth_for_del_cc)
        #4.
        save_pth_for_yolo_box = f"./{video_name}/yolo_npys"
        if not os.path.exists(save_pth_for_yolo_box): os.makedirs(save_pth_for_yolo_box)



    rgb_cap = cv2.VideoCapture(source)
    if not rgb_cap.isOpened():
        print(f"Fail to open {source}")
        os._exit(0)
    else:
        fps = int(rgb_cap.get(cv2.CAP_PROP_FPS))
        width, height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rgb_cap.release()

        # 創建資料夾
        save_path_for_ghost_detect_dict = f"./{video_name}/dict"
        if not os.path.exists(save_path_for_ghost_detect_dict): os.makedirs(save_path_for_ghost_detect_dict)

        # 初始化 dict_for_ghost_detect
        dict_for_ghost_detect = {}
        with lock_for_dict:
            with open(f"{save_path_for_ghost_detect_dict}/{video_name}.json", "w") as init_dict:
                json.dump(dict_for_ghost_detect, init_dict)

        if debug_mode:

            # if not os.path.exists(f"{video_name}/debug/aft_lightweight_denoise"): os.makedirs(f"{video_name}/debug/aft_lightweight_denoise")
            
            # 設定輸出的Mask影片
            video_wrt = cv2.VideoWriter(f"./{video_name}/{video_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    save_img = not nosave and not source.endswith('.txt') # save inference images

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if is_url and is_file: source = check_file(source) # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride) # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        '''
            type of path:    str
            type of im:      np.array with shape CHW
            type of im0s:    np.array with shape HWC
            type of vid_cap: cv2.VideoCapture
            type of s:       str
        '''

        if not run_yolo_only:
            # 存RGB影像給Tracking調用
            now_rgb_frame = im0s.copy()
            rgb_frames.put(now_rgb_frame, block=True) 
        
        bgs_frame = bgs_frames.get()
        bgs_frame_with_labels, valid_labels, stats = our_funcs.lightweight_denoise_for_mp_update(bgs_frame, debug_mode=debug_mode, save_path=f"{video_name}", save_name=f"{frame_idx}", wh_area_threshold=light_wh_area_th)
        valid_labels = set(valid_labels)
        
        # if debug_mode: cv2.imwrite(f"{video_name}/debug/aft_lightweight_denoise/{frame_idx}.png", cv2.cvtColor(bgs_frame, cv2.COLOR_GRAY2BGR))

        with dt[0]:

            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float() # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:

            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)          

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred): # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det): # 進到len(det)，代表畫面中存在物件

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) # HWC
                masks_cpu = reversed(masks.cpu()) # type(masks_cpu) 為 tensor
                # num_objs = masks_cpu.shape[0] # torch.tensor.size = (C, H, W)

                '''
                    masks[i] 代表偵測到的第i個物體的mask
                    masks_cpu.shape[0] 代表畫面中有幾個物體
                '''

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                mcolors = [colors(int(cls), True) for cls in det[:, 5]] # [1] 每個類別一種顏色
                # mcolors = [colors(int(obj_idx), True) for obj_idx in range(num_objs)] # [2] 每個物件一種顏色
                
                '''
                    1. im_masks 為原圖套上Mask之後的結果, im_masks 的shape和原圖"不同", type(im_masks) = nd.array
                    2. im_masks 為Padding後的照片，上下會有Padding區域
                    3. annotator.im()是原圖套上mask的圖片, 尚未補上BBOX，但是大小已經和原圖相同
                    4. im[i] is in cuda
                
                    做了哪些修改?
                    a. 原本是```im_masks = plot_masks(im[i], masks, mcolors)```, 改成```our_plot_masks```.            
                       our_plot_masks 多回傳一個參數 masks_color_summand = 每個Instance為彩色的 Mask, 經過 Padding 且尚未消除                    
                '''

                # im_masks, masks_color_summand = our_plot_masks(im[i], masks, mcolors) # image with masks shape(imh,imw,3)
                im_masks, _ = our_plot_masks(im[i], masks, mcolors) # image with masks shape(imh,imw,3)

                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape) # scale to original h, w

                # 將每張Masks的Padding部分刪除
                masks_cpu = our_prcoess_bin_masks(masks_cpu, im.shape[2:], im0.shape) # now, masks_cpu is np.ndarray

                plot_labels = valid_labels.copy()
                arr_for_ghost_detect = np.array([]) # arr_for_ghost_detect 就是yolo_bbox
                cc_del = np.array([])

                for obj_idx, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                    # Set up obj info
                    # obj_cls = names[int(cls)]                        
                    # ul, lr = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    arr_for_ghost_detect = np.append( arr_for_ghost_detect, [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])] )

                    obj_mask = masks_cpu[obj_idx]                    
                    # yolo_white_pixels = np.count_nonzero( obj_mask )

                    # bbox info: ul, lr = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    yolo_ul, yolo_lr = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                    for valid_label in valid_labels:
                        cc_bbox = stats[valid_label] # x, y, w, h
                        cc_ul, cc_lr = (cc_bbox[0], cc_bbox[1]), (cc_bbox[0]+cc_bbox[2], cc_bbox[1]+cc_bbox[3])

                        intersect_ul, intersect_lr = ( max(yolo_ul[0], cc_ul[0]), max(yolo_ul[1], cc_ul[1]) ), ( min(yolo_lr[0], cc_lr[0]), min(yolo_lr[1], cc_lr[1]) )
                        intersect_area = max(intersect_lr[0]-intersect_ul[0], 0)*max(intersect_lr[1]-intersect_ul[1], 0) # w*h

                        # yolo偵測物與連通域進行配對
                        if (not intersect_area==0) and (valid_label in plot_labels):
                            # plot_labels.discard(valid_label)
                            plot_labels -= {valid_label}
                            cc_del = np.append(cc_del, [cc_bbox[0], cc_bbox[1], cc_bbox[2], cc_bbox[3]])

                    # if save_txt: # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh) # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img: # Add bbox to image
                        c = int(cls) # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # type(label) 為 string, label = person 0.94

                        # 加入Yolo_BBox訊息
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                denoise_mask = np.zeros((height, width, 3), np.uint8)
                for plot_label in plot_labels:
                    denoise_mask[bgs_frame_with_labels==plot_label] = (255, 255, 255)
                # 灰階 -> 膨脹
                denoise_mask = cv2.dilate( cv2.cvtColor(denoise_mask, cv2.COLOR_BGR2GRAY), kernel, iterations=2)
                denoise_mask, cc_bboxes = our_funcs.denoise(denoise_mask, debug_mode=debug_mode, save_path=f"{video_name}/debug", save_name=f"{frame_idx}", wh_area_threshold=wh_area_th)

                if debug_mode: video_wrt.write(denoise_mask)

                '''
                    cc BBox Info.: 
                        1. [x, y, w, h]
                    ----------  
                    yolo bboxes info.:
                        1. yolo_bboxes.shape = (num_objs, 4)
                        2. yolo_bboxes[0] = [ ul_x, ul_y, lr_x, lr_y ]                
                '''

                # 將yolo_bbox寫入dict
                arr_for_ghost_detect = arr_for_ghost_detect.reshape((-1, 4)) # arr_for_ghost_detect 就是yolo_bbox
                dict_for_ghost_detect[frame_idx] = arr_for_ghost_detect.tolist()

                # 處理連通域相關的BBox
                cc_del = cc_del.reshape((-1, 4))          

            else: # yolo沒有偵測到物件

                bgs_frame, cc_bboxes = our_funcs.denoise(bgs_frame, debug_mode=debug_mode, save_path=f"{video_name}/debug", save_name=f"{frame_idx}", wh_area_threshold=wh_area_th)
                if debug_mode: video_wrt.write(bgs_frame)

                # 將yolo_bbox寫入dict
                arr_for_ghost_detect = np.array([]) # yolo沒有偵測到物件
                dict_for_ghost_detect[frame_idx] = arr_for_ghost_detect.tolist()

                # 處理連通域相關的BBox
                cc_del = np.array([]) # 沒有偵測到任何物件，不會有刪除連通域的問題

            if run_yolo_only:
                # np.save(f"{save_pth_for_yolo_box}/{frame_idx}.npy", arr_for_ghost_detect)
                # np.save(f"{save_pth_for_res_cc}/{frame_idx}.npy", cc_bboxes)
                # np.save(f"{save_pth_for_del_cc}/{frame_idx}.npy", cc_del)
                pass
            else:
                cc_bboxes_info.put(cc_bboxes, block=True)
                cc_del_info.put(cc_del, block=True)  

            # Stream results
            im0 = annotator.result()

            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # else:  # stream
                        #     fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 每十幀Update一次 dict_for_ghost_detect
        if (frame_idx+1)%10==0:
            with lock_for_dict:
                with open(f"{save_path_for_ghost_detect_dict}/{video_name}.json", "w") as updated_dict:
                    json.dump(dict_for_ghost_detect, updated_dict)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

from torchreid.utils import FeatureExtractor
def tracking(args, video_name, save_path_for_dict, frame_count, rgb_frames, cc_bboxes_info, cc_del_info, lock_for_dict):

    extractor = FeatureExtractor(
        model_name='resnext101_32x8d',
        model_path='log/resnext101/model/model.pth.tar-60',
        device='cuda'
    )

    check_yolo_for_ghost = [30, 300, 1800, 3600, 5400]

    # Initialize Tracking List
    tracking_list = []
    cover_by_yolo = []

    # 初始化上一幀
    previous_rgb_frame = None

    for frame_idx in range(frame_count):

        # Get the value from Queue
        now_rgb_frame = rgb_frames.get()
        interest_bboxes = cc_bboxes_info.get()
        del_interest_bboxed = np.unique(cc_del_info.get(), axis=0)

        for idx in reversed(range(len(tracking_list))):
            x = tracking_list[idx]
            if check_coverby_delcc(del_interest_bboxed, x["bbox"]) == True:
                if check_in_coverbyyolo(cover_by_yolo, x["bbox"]) == False:
                    cover_by_yolo.append(x)
                del tracking_list[idx]

        for idx in reversed(range(len(cover_by_yolo))):
            y = cover_by_yolo[idx]
            if check_coverby_delcc(del_interest_bboxed, y["bbox"]) == False:
                tracking_list.append(y)
                del cover_by_yolo[idx]

        wait_to_add = [] # wait_to_add 用來儲存加入清單之物體

        # create cost_matrix
        n, m, k= len(interest_bboxes), len(tracking_list), len(cover_by_yolo)

        if m == 0: # tracking list is empty                
            for i, y_box in enumerate(interest_bboxes):
                wait_to_add = add_trackinglist(wait_to_add, y_box, frame_idx)
        elif m != 0 and n != 0:
            cost_matrix = np.zeros((n, m))
            for i, y_box in enumerate(interest_bboxes): # y
                y_area = y_box[2]*y_box[3]
                
                for j, x_box in enumerate(tracking_list):    # x
                    # get iou
                    iou_score = our_funcs.iou(y_box, x_box["bbox"])

                    # get difference of box size
                    x_area = x_box["bbox"][2]*x_box["bbox"][3]
                    size_score = abs(y_area - x_area) / max(y_area, x_area)

                    # get similarity score
                    sim_score = get_similarity(extractor, y_box, now_rgb_frame, x_box["bbox"], previous_rgb_frame)

                    # 計算總權重
                    total_score = iou_score + (1 - size_score) + sim_score

                    cost_matrix[i][j] = total_score

            max_cost = np.max(cost_matrix) 
            cost_matrix = max_cost - cost_matrix # 原本是越大越好（因為相似度、iou越大越好），要改成越小越好
            match = hungarian_algorithm(cost_matrix)

            for i, y_box in enumerate(interest_bboxes): 
                need_update = True
                if match[i] == -1:        # y比x多，沒有配到x -> 新物體
                    wait_to_add = add_trackinglist(wait_to_add, y_box, frame_idx)
                else:                       # 有配到，檢查有沒有過三關
                    if our_funcs.iou(y_box, tracking_list[match[i]]["bbox"]) > args.high_iou_th and get_similarity(extractor, y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], previous_rgb_frame) > args.high_iou_simi_th:
                        pass

                    elif (our_funcs.iou(y_box, tracking_list[match[i]]["bbox"]) < args.overlap_th) or check_sizediff(y_box, tracking_list[match[i]]["bbox"]) \
                            or (get_similarity(extractor, y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], previous_rgb_frame) < args.similarity_th):
                        # 沒過--> 新物體
                        wait_to_add = add_trackinglist(wait_to_add, y_box, frame_idx)
                        need_update = False

                    else:   # 有配到且通過判斷 --> 更新
                        pass

                    if need_update:
                        if tracking_list[match[i]]["state"] == -2:
                            continue
                        tracking_list[match[i]] = update_trackinglist(tracking_list[match[i]], y_box)

                        if tracking_list[match[i]]["duration"] >= args.stay_up_th and tracking_list[match[i]]["moved_dist"] < args.moved_th:
                            was_in_yolo = 0

                            for t in check_yolo_for_ghost:
                                check_yolo_idx = tracking_list[match[i]]["frame_idx"] - t
                                if check_yolo_idx < 0:
                                    check_yolo_idx = 0

                                while True:
                                    with lock_for_dict:

                                        # 必須保存先前的Yolo BBox
                                        with open( f"{save_path_for_dict}/{video_name}.json" , 'r') as f:
                                            data = json.load(f)
                                        
                                        if f"{check_yolo_idx}" in data:
                                            pre_yolo_bbox = np.array(data[f"{check_yolo_idx}"])
                                            break

                                    time.sleep(0.5)

                                sizediff_with_yolo = check_in_yolo(pre_yolo_bbox, tracking_list[match[i]]["appear_bbox"]) # if == 0 : not in yolo, else is in yolo and return size diff
                                if sizediff_with_yolo > 0 and sizediff_with_yolo < args.sizediff_yolo_th:
                                    was_in_yolo += 1

                            if was_in_yolo >= 4 :
                                tracking_list[match[i]]["duration"] = 0
        
                            else:
                                print("garbage/{}".format(tracking_list[match[i]]["frame_idx"]))
                                tracking_list[match[i]]["is_garbage"] = True
                                tracking_list[match[i]]["state"] = -2

                        if tracking_list[match[i]]["duration"] > args.stay_up_th / 2:
                            tracking_list[match[i]]["moved_dist"] = 0

        remove_idx = []

        # 新增
        for add_obj in wait_to_add:
            tracking_list.append(add_obj)

        for z_idx, z in enumerate(tracking_list): 

            if z["ref"] == True or z["state"] == -2:
                continue
            z["state"] = 1
            # 篩掉那些被覆蓋住的
            be_covered = False

            ### covered by other tracking box
            for x_idx, x in enumerate(tracking_list):
                if x_idx == z_idx:
                    continue
                if x["state"] == -1 or x["state"] == -2 or (x["no_pair_counter"]+1 >= args.delete_cnt and x["state"] != -2):
                    continue
                
                cover_area = check_covered(z["bbox"], x["bbox"])
                if cover_area > args.covered_th:
                    z["be_covered"] += 1
                    z["state"] = -1
                    be_covered = True
                    break

            if be_covered and z["be_covered"] < args.be_covered_time_th:
                continue

            z["no_pair_counter"] += 1
            if z["no_pair_counter"] >= args.delete_cnt and not z["is_garbage"]: # 連續delete_cnt幀沒配到前景，且非垃圾，刪除
                remove_idx.append(z_idx)
                continue

            if z["is_garbage"] == True:
                z["state"] = -2


        # 更新 Tracking List 各個物件的 ref 等等參數
        # 刪除
        for r_idx in remove_idx[::-1]:
            # remove_idx 中的物件必為 Tracking List 的子集合 
            del tracking_list[r_idx]

        # 更新參數
        for obj_idx, obj in enumerate(tracking_list):
            tracking_list[obj_idx]["ref"] = False

        # 設置上一幀
        previous_rgb_frame = now_rgb_frame.copy()

        if len(tracking_list) >= args.reset_cnt: 
            print("Too many box, reset tracking list")
            tracking_list.clear()

# def parse_opt(): # For yolo
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
#     parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt

def main():

    check_requirements(exclude=('tensorboard', 'thop'))

    # user config
    use_table = False 

    frame_interval = 3 # 介於1~3之間。設為4，BGS的雜訊會太大

    run_yolo_only = True

    # 影片10在不跳幀的狀況下，跑了快40分鐘

    if use_table: # 用vid_table

        vid_table = [
            "0_常見機車場景.mp4",
            "1_常見機車場景.mp4",
            "2_雨天迴轉.mp4",
            "3_實驗室外走廊.mp4",
            "4_工三外_少人.mp4",
            "5_工三外_多人.mp4",
            "6_短影片_修改第一幀.mp4",
            "7_短影片_修改第一幀.mp4",
            "8_短影片.mp4",
            "9_短影片_修改第一幀.mp4",
            "10_純雨天無垃圾.mp4",
            "11_雨天中間距離丟垃圾.mp4",
            "12_雨天近距離丟垃圾.mp4",
            "13_雨天遠距離丟垃圾.mp4", # Bad
            "14_人擋住垃圾.mp4",
            "15_逗留.mp4",
            "16_反覆提起垃圾.mp4",
            "17_實驗室丟背包.mp4",
            "18_影片.mp4",
            "19_影片.mp4",
            "20_尼斯湖.mp4",
            "21_浴室.mp4",
            "22_室外拍傘.mp4"
        ]
        
        vid_need_process = []
        assert not len(vid_need_process)==0

    else: # 不用vid_table
        vid_need_process = natsorted(os.listdir("./data/our_videos/rgb/"))

    for vid_id in vid_need_process:

        start_time = time.time()
        
        # path2rgb指向原始的影片
        path2rgb = f"./data/our_videos/rgb/{vid_table[vid_id]}" if use_table else f"./data/our_videos/rgb/{vid_id}"

        video_name = path2rgb.split('/')[-1]
        video_name = video_name.split('.')[0]

        if not os.path.exists(f"./{video_name}"): os.makedirs(f"./{video_name}")

        # our_funcs.modify_video(path2rgb, save_name=f"{video_name}_resize.mp4", save_path=f"./{video_name}", resize=False, resize_height=-1, resize_width=-1, frame_interval=frame_interval)        
        # source = f"./{video_name}/{video_name}_resize.mp4"

        source = path2rgb

        rgb_cap = cv2.VideoCapture(source)
        frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rgb_cap.release()

        # Create Queue
        bgs_frames = Queue(maxsize=100)

        if run_yolo_only:
            rgb_frames = None
            cc_bboxes_info = None
            cc_del_info = None
        else:
            rgb_frames = Queue(maxsize=100)
            cc_bboxes_info = Queue(maxsize=100)
            cc_del_info = Queue(maxsize=100)
        
        # Create Lock to Solve Sync. Prob.
        lock_for_dict = Lock()

        # Setup the Processes
        bgs_process = Process(target=our_funcs.bgs_generator_for_mp, args=(source, bgs_frames))
        yolo_process = Process(target=run, args=(bgs_frames, rgb_frames, cc_bboxes_info, cc_del_info, lock_for_dict, run_yolo_only, ), kwargs={'source':source}) 
        if not run_yolo_only:
            save_path_for_dict = f"{video_name}/dict"
            tracking_process = Process(target=tracking, args=(tracking_config(), video_name, save_path_for_dict, frame_count, rgb_frames, cc_bboxes_info, cc_del_info, lock_for_dict))

        # 當run_yolo_only==False時，才要跑tracking_process
        bgs_process.start()
        yolo_process.start()
        if not run_yolo_only:
            tracking_process.start()

        bgs_process.join()
        yolo_process.join()
        if not run_yolo_only:
            tracking_process.join()

        end_time = time.time()
        print(f"Execution time: {round(end_time-start_time, 4)}")

if __name__ == "__main__":
    main()