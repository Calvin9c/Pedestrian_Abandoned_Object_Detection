import sys
import os
from natsort import natsorted

import numpy as np
import cv2
import pybgs as bgs

def folder_initialization(cfg):

    os.makedirs(cfg['video_name'], exist_ok=True)
    os.makedirs(cfg['save_path_for_dict'], exist_ok=True)

    if cfg['run_yolo_only']:
        os.makedirs(cfg['save_pth_for_res_cc'], exist_ok=True)
        os.makedirs(cfg['save_pth_for_del_cc'], exist_ok=True)
        os.makedirs(cfg['save_pth_for_yolo_box'], exist_ok=True)
    
    if cfg['debug_mode']:
        os.makedirs(cfg['denoise_fst_proc'], exist_ok=True)
        os.makedirs(cfg['denoise_sec_proc'], exist_ok=True)
        os.makedirs(cfg['denoise_trd_proc'], exist_ok=True)
        os.makedirs(cfg['del_intersect_area'], exist_ok=True)

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

def bgs_proc(path2rgb, frame_interval=2, save_path="./", save_name="bgs_result.mp4"):
    rgb_cap = cv2.VideoCapture(path2rgb)

    if rgb_cap.isOpened():
        h, w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        size = (w, h)
        fps        = rgb_cap.get(cv2.CAP_PROP_FPS)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        videoWrite = cv2.VideoWriter( os.path.join(save_path, save_name) , fourcc, fps, size)
    else:
        print(f"Fail to conduct BGS Process.")
        os._exit(0)


    algorithm = bgs.ViBe()
    while True:

        frame_cnter = int(rgb_cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"frame_cnter: {frame_cnter}")
        rval, frame = rgb_cap.read()

        if rval:

            if frame_cnter % frame_interval == 0:

                img_output = algorithm.apply(frame)
                img_bgmodel = algorithm.getBackgroundModel()

                videoWrite.write(cv2.cvtColor(img_output, cv2.COLOR_GRAY2BGR))

        else: break
    rgb_cap.release()
    videoWrite.release ()

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

    if strt_idx < cc_bboxes.shape[0]:
        
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

def bgs_generator_old(
        
        # input_video
        path2rgb,
        frame_interval,

        # buf
        buf_bgs_frames, 
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

    rgb_cap = cv2.VideoCapture(path2rgb)

    if not rgb_cap.isOpened():
        print(f"Fail to conduct BGS Process.")
        os._exit(0)

    algorithm = bgs.ViBe()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:

        frame_cnter = int(rgb_cap.get(cv2.CAP_PROP_POS_FRAMES))
        rval, frame = rgb_cap.read()

        if rval:

            if frame_cnter % frame_interval == 0:
                
                frame_idx = frame_cnter // frame_interval

                img_output = algorithm.apply(frame)
                img_bgmodel = algorithm.getBackgroundModel()

                # save_path 設為 f"{vid_name}", save_name 設為 frame_idx
                bgs_frame, img_with_labels, val_lbls, inval_lbls, cc_bboxes = denoise(
                                                                                img_output, 
                                                                                kernel, 
                                                                                debug_mode, 
                                                                                save_path, 
                                                                                frame_idx, 
                                                                                sec_proc_area_th, 
                                                                                trd_proc_area_th
                                                                            )

                buf_bgs_frames.put( bgs_frame, block=True )
                buf_img_with_lbls.put( img_with_labels, block=True )
                buf_val_lbls.put( val_lbls, block=True )
                buf_inval_lbls.put( inval_lbls, block=True )
                buf_cc_bboxes.put( cc_bboxes, block=True )

        else: break
    rgb_cap.release()

def bgs_generator(
        
        # input_video
        path2rgb,
        frame_interval,

        # buf
        buf_bgs_frames,

        # Debug Setting
        debug_mode,
        save_path # vid_name
    ):

    rgb_cap = cv2.VideoCapture(path2rgb)

    if not rgb_cap.isOpened():
        print(f"Fail to conduct BGS Process.")
        os._exit(0)

    algorithm = bgs.ViBe()

    if debug_mode:
        videoWrite = cv2.VideoWriter(
            f"{save_path}/{save_path}_BGS.mp4" ,                                                        # save_name
            cv2.VideoWriter_fourcc(*"mp4v"),                                                            # fourcc
            int(rgb_cap.get(cv2.CAP_PROP_FPS)),                                                         # FPS
            ( int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) ) # size = (w, h)
        )

    while True:

        frame_cnter = int(rgb_cap.get(cv2.CAP_PROP_POS_FRAMES))
        rval, frame = rgb_cap.read()

        if rval:

            if frame_cnter % frame_interval == 0:
                
                frame_idx = frame_cnter // frame_interval

                img_output  = algorithm.apply(frame)
                img_bgmodel = algorithm.getBackgroundModel()

                buf_bgs_frames.put( img_output, block=True )

                if debug_mode:
                    videoWrite.write(cv2.cvtColor(img_output, cv2.COLOR_GRAY2BGR))

        else: break

    rgb_cap.release()
    
    if debug_mode: videoWrite.release()

def denoise(img, kernel, debug_mode=False, save_path=None, save_name=None, sec_proc_area_th=150, trd_proc_area_th=800):

    # save_path 設為 f"{vid_name}"
    # save_name 設為 frame_idx

    if not len(img.shape)==2: # input 需要是單通道圖片
        print("The input of denoise function should be single channel! Please checkout the function input.") # 輸入非單通道圖片
        sys.exit(1)

    # first process: 二值化
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if debug_mode: 
        cv2.imwrite(f"{save_path}/debug/denoise/first_process/{save_name}.png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    # second_process: 刪除過小的白色區域
    # bgs_frame 為單通道
    bgs_frame, _, _, _, _ = my_connectedComponentsWithStats(img, sec_proc_area_th, debug_mode, f"{save_path}/debug/denoise/second_process", save_name)
    bgs_frame = cv2.dilate(bgs_frame, kernel, iterations=3)

    # third_process: 刪除大的雜訊
    bgs_frame, img_with_labels, val_lbls, inval_lbls, cc_bboxes = my_connectedComponentsWithStats(bgs_frame, trd_proc_area_th, debug_mode, f"{save_path}/debug/denoise/third_process", save_name)

    return bgs_frame, img_with_labels, set(val_lbls), inval_lbls, cc_bboxes

from concurrent.futures import ThreadPoolExecutor
def process_masks_parallel_merge(AMask, BMasks):
    def process_single_frame(AMask, BMask):
        AMaskCopy = np.copy(AMask)
        _, BMask  = cv2.threshold(BMask, 0, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_xor(AMaskCopy, cv2.bitwise_and(AMaskCopy, BMask))

    def pairwise_merge(results):
        with ThreadPoolExecutor() as executor:
            merged_results = list(executor.map(
                                    lambda pair: cv2.bitwise_and(pair[0], pair[1]), 
                                                 [results[i:i+2] for i in range(0, len(results), 2)]
                             ))
        return merged_results

    # Step 1: Parallel processing of each BMask
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda BMask: process_single_frame(AMask, BMask), BMasks))

    # Step 2: Pairwise bitwise_and to merge results (parallelized)
    while len(results) > 1:
        if len(results) % 2 != 0:  # For odd number of elements, keep the last one for the next round
            results = pairwise_merge(results[:-1]) + [results[-1]]
        else:
            results = pairwise_merge(results)

    return results[0] if results else None




















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