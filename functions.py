import sys
import os
from natsort import natsorted

import numpy as np
import cv2
import pybgs as bgs

def create_debug_folders(vid_name):
    if not os.path.exists(f"{vid_name}/debug/denoise/first_process"):
        os.makedirs(f"{vid_name}/debug/denoise/first_process")
    if not os.path.exists(f"{vid_name}/debug/denoise/second_process"):
        os.makedirs(f"{vid_name}/debug/denoise/second_process")
    if not os.path.exists(f"{vid_name}/debug/denoise/third_process"):
        os.makedirs(f"{vid_name}/debug/denoise/third_process")

from tqdm import tqdm
def modify_video(path2video, save_name=None, save_path="./result_modify_video", resize=False, resize_height=-1, resize_width=-1, frame_interval=1):

    if not os.path.exists(path2video):
        print( "Fail to open "+path2video )
        os._exit(0)

    if not os.path.exists(save_path): os.makedirs(save_path)

    if save_name==None:
      save_name = path2video.split('/')[-1]

    cap = cv2.VideoCapture(path2video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if resize:
        assert not (resize_height==-1 and resize_width==-1), "Please set up resize_height, resize_width."
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if resize_height==-1 else resize_height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if resize_width==-1 else resize_width
    else:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    modified_vid = cv2.VideoWriter(f"{save_path}/{save_name}", fourcc, fps, (width, height))
    
    progress = tqdm(total=frame_count)
    for frame_idx in range(frame_count):

        rval, frame = cap.read()
        if not rval:
            print(f"Fail to read frame {frame_idx}.\nKill the process.")
            os._exit(0)
        
        if not frame_idx%frame_interval==0: 
            progress.update(1)
            continue

        if resize:
            frame = cv2.resize(frame, (width, height))
        
        modified_vid.write(frame)
        progress.update(1)
    
    modified_vid.release()

def my_connectedComponentsWithStats(img, area_threshold, debug_mode=False, save_path=None, save_name=None):

    # ---------- 處理 input ---------- # 

    if len(img.shape) == 2: # img 應為單通道
        height, width = img.shape
    else:
        print("Input of my_connectedComponentsWithStats should be single channel.")
        sys.exit(1)

    if debug_mode:

        assert not(save_path==None or save_name==None)

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        debug_img = np.zeros((height, width, 3), np.uint8)

    # ---------- ---------- ---------- #

    num_labels, img_with_labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    inval_lbls, val_lbls = [], []
    strt_idx = 0
    while not (cc_bboxes[strt_idx][0]==0 and cc_bboxes[strt_idx][1]==0 and cc_bboxes[strt_idx][2]==width and cc_bboxes[strt_idx][3]==height):

        inval_lbls.append(strt_idx)

        strt_idx += 1

        if strt_idx == cc_bboxes.shape[0]: break
    
    while cc_bboxes[strt_idx][0]==0 and cc_bboxes[strt_idx][1]==0 and cc_bboxes[strt_idx][2]==width and cc_bboxes[strt_idx][3]==height:

        inval_lbls.append(strt_idx)

        strt_idx += 1

        if strt_idx == cc_bboxes.shape[0]: break

    denoise_img = np.zeros((height, width), np.uint8)
    for component_label in range(strt_idx, num_labels):
        
        if cc_bboxes[component_label][4]>=area_threshold:

            denoise_img[img_with_labels==component_label] = 255
            val_lbls.append(component_label)
            
            if debug_mode:
                
                debug_img[img_with_labels==component_label] = (255, 255, 255)

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]

                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

        else:
            inval_lbls.append(component_label)

    if debug_mode: 
        cv2.imwrite(f"{save_path}/{save_name}.png", debug_img)

    return denoise_img, img_with_labels, val_lbls, inval_lbls, cc_bboxes

def bgs_generator(
        
        # pth2vid
        path2rgb,

        # buf
        buf_img_with_lbls,
        buf_val_lbls,
        buf_inval_lbls, 
        buf_cc_bboxes,

        # Threshold
        sec_proc_area_th,
        trd_proc_area_th,
        debug_mode,
        save_path # vid_name
    ):

    cap = cv2.VideoCapture(path2rgb)
    if not cap.isOpened():
        print(f"Fail to open {path2rgb}")
        os._exit(0)

    algorithm = bgs.ViBe()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while True:

        rval, frame = cap.read()

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if rval:
            img_output = algorithm.apply(frame)
            img_bgmodel = algorithm.getBackgroundModel()

            # save_path 設為 f"{vid_name}", save_name 設為 frame_idx
            img_with_labels, val_lbls, inval_lbls, cc_bboxes = denoise(
                                                                    img_output, 
                                                                    kernel, 
                                                                    debug_mode, 
                                                                    save_path, 
                                                                    frame_idx, 
                                                                    sec_proc_area_th, 
                                                                    trd_proc_area_th
                                                                )

            buf_img_with_lbls.put( img_with_labels, block=True )
            buf_val_lbls.put( val_lbls, block=True )
            buf_inval_lbls.put( inval_lbls, block=True )
            buf_cc_bboxes.put( cc_bboxes, block=True )

        else: break
    cap.release()

def denoise(img, kernel, debug_mode=False, save_path=None, save_name=None, sec_proc_area_th=150, trd_proc_area_th=800):

    # save_path 設為 f"{vid_name}"
    # save_name 設為 frame_idx

    if not len(img.shape)==2: # input 需要是單通道圖片
        print("The input of denoise function should be single channel! Please checkout the function input.") # 輸入非單通道圖片
        sys.exit(1)

    # first process: 模糊、二值化
    # img = cv2.medianBlur(img, 3)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if debug_mode: 
        cv2.imwrite(f"{save_path}/debug/denoise/first_process/{save_name}.png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    # second_process: 刪除過小的白色區域
    # bgs_frame 為單通道
    bgs_frame, _, _, _, _ = my_connectedComponentsWithStats(img, sec_proc_area_th, debug_mode, f"{save_path}/debug/denoise/second_process", save_name)
    bgs_frame = cv2.dilate(bgs_frame, kernel, iterations=3)

    # third_process: 刪除大的雜訊
    _, img_with_labels, val_lbls, inval_lbls, cc_bboxes = my_connectedComponentsWithStats(bgs_frame, trd_proc_area_th, debug_mode, f"{save_path}/debug/denoise/third_process", save_name)

    return img_with_labels, set(val_lbls), inval_lbls, cc_bboxes






















# Tracking
def dilate_bbox(bbox, img_height, img_width, dilate_ratio=0.1):
    
    ul_x, ul_y = max(int(bbox[0]-bbox[2]*0.1), 0), max(int(bbox[1]-bbox[3]*0.1), 0)
    lr_x, lr_y = min(int(bbox[0]+(1.1)*bbox[2]-1), img_width), min(int(bbox[1]+(1.1)*bbox[3]), img_height)

    return (ul_x, ul_y), (lr_x, lr_y)

def iou(a, b):
    # x, y, w,  h,  s
    area_a = a[2]*a[3]
    area_b = b[2]*b[3]

    w = min(b[0]+b[2], a[0]+a[2]) - max(a[0], b[0])
    h = min(b[1]+b[3], a[1]+a[3]) - max(a[1], b[1])

    if w<=0 or h<=0:
        return 0
    
    area_c = w*h

    return area_c/(area_a+area_b-area_c)