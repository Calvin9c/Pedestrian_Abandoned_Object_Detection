import sys
import os
from natsort import natsorted

import numpy as np
import cv2
import pybgs as bgs

def create_debug_folders(vid_name):

    '''
        lightweight_denoise 分成三流程
        1. 模糊、二值化
        2. 消除小白點
        3. 兩次膨脹
    '''
    if not os.path.exists(f"{vid_name}/debug/lightweight_denoise/first_process"):
        os.makedirs(f"{vid_name}/debug/lightweight_denoise/first_process")
    if not os.path.exists(f"{vid_name}/debug/lightweight_denoise/second_process"):
        os.makedirs(f"{vid_name}/debug/lightweight_denoise/second_process")
    if not os.path.exists(f"{vid_name}/debug/lightweight_denoise/third_process"):
        os.makedirs(f"{vid_name}/debug/lightweight_denoise/third_process")

    '''
        denoise 分成兩流程
        1. 模糊、二值化、一次膨脹
        2. 刪除大面積雜訊
    '''
    if not os.path.exists(f"{vid_name}/debug/denoise/first_process"):
        os.makedirs(f"{vid_name}/debug/denoise/first_process")
    if not os.path.exists(f"{vid_name}/debug/denoise/second_process"):
        os.makedirs(f"{vid_name}/debug/denoise/second_process")

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








# predict.py, predict_update.py
# 1. lightweight_denoise
# 2. denoise_if_detect_obj / denoise

def lightweight_denoise(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=150):

    # save_path 設為 f"{vid_name}"
    # save_name 設為 frame_idx

    # 輸入為三通道的圖片；輸出為單通道的圖片
    # 共經過first process, second process, third process 三流程

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        debug_img = np.zeros((height, width, 3), np.uint8)

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

    # [ first process ] 二值化
    # mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # [ second_process ] 刪除過小的白色區域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 stats 異常區域
    remove_idx, start = [], 0
    while not(stats[start][0]==0 and stats[start][1]==0 and stats[start][2]==width and stats[start][3]==height):

        remove_idx.append(start)

        start += 1

        if start==stats.shape[0]: break

    # 移除 stats 黑色區域
    while stats[start][0]==0 and stats[start][1]==0 and stats[start][2]==width and stats[start][3]==height:

        remove_idx.append(start)

        start += 1

        if start == stats.shape[0]: break

    ret_mask = np.zeros((height, width), np.uint8)
    for component_label in range(start, num_labels):

        if stats[component_label][4] >= wh_area_threshold:

            ret_mask[labels==component_label] = 255

            if debug_mode:
                debug_bbox = stats[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]
                
                debug_img[labels==component_label] = (255, 255, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

    # [ third_process ] 膨脹*2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    ret_mask = cv2.dilate(ret_mask, kernel, iterations=2)

    if debug_mode: 
        # first_process result
        cv2.imwrite(f"{save_path}/debug/lightweight_denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # second_process result
        cv2.imwrite(f"{save_path}/debug/lightweight_denoise/second_process/{save_name}.png", debug_img)
        # third_process result
        cv2.imwrite(f"{save_path}/debug/lightweight_denoise/third_process/{save_name}.png", cv2.cvtColor(ret_mask, cv2.COLOR_GRAY2BGR))

    return ret_mask # ret_mask 為單通道

def bgs_generator_with_denoise(path2rgb, queue, wh_area_threshold, debug_mode, save_path):

    cap = cv2.VideoCapture(path2rgb)
    if not cap.isOpened():
        print(f"Fail to open {path2rgb}")
        os._exit(0)

    algorithm = bgs.ViBe()

    while True:

        rval, frame = cap.read()

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if rval:
 
            img_output = algorithm.apply(frame)
            img_bgmodel = algorithm.getBackgroundModel()

            # save_path 設為 f"{vid_name}", save_name 設為 frame_idx
            img_output = lightweight_denoise(img_output, debug_mode, save_path, save_name=frame_idx, wh_area_threshold=wh_area_threshold)

            queue.put(img_output, block=True)
        else: break
    cap.release()

def denoise_if_detect_obj(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=600):

    # 輸入為單通道

    # save_path 設為 vid_name
    # save_name 設為 frame_idx

    try: # input: mask 是二通道的圖片
        height, width = mask.shape
    except: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        debug_img = np.zeros((height, width, 3), np.uint8)

    # first_process: 模糊、二值化、膨脹
    # mask = cv2.medianBlur(mask, 3)
    # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    mask = cv2.dilate(mask, kernel, iterations=1)

    # second_process: 刪除過小的白色區域
    # cc_bboxes: x, y, w, h
    num_labels, labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 中異常區域
    start = 0
    remove_labels = []  
    while not (cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height):

        remove_labels.append(start)

        start += 1

        if start==cc_bboxes.shape[0]: break

    while cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height:

        remove_labels.append(start)

        start += 1

        if start == cc_bboxes.shape[0]: break
    
    reserve_labels = []
    for component_label in range(start, num_labels):

        if cc_bboxes[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            reserve_labels.append(component_label)

            if debug_mode:

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]
                
                debug_img[labels==component_label] = (255, 255, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

        else: remove_labels.append(component_label)

    if debug_mode: 
        # first_process
        cv2.imwrite(f"{save_path}/debug/denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # second_process
        cv2.imwrite(f"{save_path}/debug/denoise/second_process/{save_name}.png", debug_img)
         
    # 回傳內容如下
    
    # [必須回傳]
    # 二次降躁前，所有連通域的bboxes: cc_bboxes
    # 被保留的連通域清單: reserve_labels
    # 後續將被刪除的連通域清單: remove_labels

    if debug_mode:
        # [debug用]
        # 帶框的三通道圖片: debug_img
        # 帶有label的二維陣列: labels
        return debug_img, labels, cc_bboxes, reserve_labels, remove_labels
    else:
        return cc_bboxes, reserve_labels, remove_labels
    
def denoise(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=600): 

    # 輸入為單通道

    # save_path 設為 vid_name / debug
    # save_name 設為 frame_idx

    try: # input: mask 是二通道的圖片
        height, width = mask.shape
    except: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        debug_img = np.zeros((height, width, 3), np.uint8)

    # first_process: 模糊、二值化、膨脹
    # mask = cv2.medianBlur(mask, 3)
    # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    mask = cv2.dilate(mask, kernel, iterations=1)

    # second_process: 刪除過小的白色區域
    num_labels, labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 中異常區域
    remove_idx, start = [], 0 
    while not (cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height):

        remove_idx.append(start)

        start+=1

        if start==cc_bboxes.shape[0]: break

    
    while cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height:

        remove_idx.append(start)

        start+=1

        if start==cc_bboxes.shape[0]: break

    for component_label in range(start, num_labels):
        
        if cc_bboxes[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            
            if debug_mode:
                
                debug_img[labels==component_label] = (255, 255, 255)

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]

                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

        else:
            remove_idx.append(component_label)

    cc_bboxes = np.delete(cc_bboxes, remove_idx, axis=0)

    if debug_mode: 
        # first_process
        cv2.imwrite(f"{save_path}/debug/denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # second_process
        cv2.imwrite(f"{save_path}/debug/denoise/second_process/{save_name}.png", debug_img)
        # 回傳三通道的ret_mask
        return debug_img, cc_bboxes
    else: 
        return cc_bboxes

# predict_update.py 刪除連通域用
def del_cc_region(args):

    yolo_bbox, component_label, cc_info = args

    yolo_ul, yolo_lr = (int(yolo_bbox[0]), int(yolo_bbox[1])), (int(yolo_bbox[2]), int(yolo_bbox[3]))
    yolo_area = (yolo_lr[0]-yolo_ul[0])*(yolo_lr[1]-yolo_ul[1]) # w*h

    cc_ul, cc_lr = (cc_info[0], cc_info[1]), (cc_info[0]+cc_info[2], cc_info[1]+cc_info[3])

    intersect_ul, intersect_lr = ( max(yolo_ul[0], cc_ul[0]), max(yolo_ul[1], cc_ul[1]) ), ( min(yolo_lr[0], cc_lr[0]), min(yolo_lr[1], cc_lr[1]) )
    intersect_area = max(intersect_lr[0]-intersect_ul[0], 0)*max(intersect_lr[1]-intersect_ul[1], 0) # w*h

    # yolo_mask為連通區域的subset
    if intersect_area/yolo_area >= 0.9: # 配對成功
        return component_label
    return None









def my_connectedComponentsWithStats(img, area_threshold, debug_mode=False, save_path=None, save_name=None):

    # ---------- 處理 input ---------- # 

    if len(img.shape) == 2: # img 應為單通道
        height, width = img.shape
    else:
        print("Input of my_connectedComponentsWithStats should be single channel.")
        sys.exit(1)

    if debug_mode:

        assert not(save_path==None or save_name=None)

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
        cv2.imwrite(f"{save_path}/debug/denoise/second_process/{save_name}.png", debug_img)

    return denoise_img, img_with_labels, val_lbls, inval_lbls, cc_bboxes












# predict_wo_update.py
def bgs_generator_for_mp(
        path2rgb,

        bgs_frames,

        wh_area_threshold,
        debug_mode,
        save_path
    ):

    cap = cv2.VideoCapture(path2rgb)
    if not cap.isOpened():
        print(f"Fail to open {path2rgb}")
        os._exit(0)

    algorithm = bgs.ViBe()

    while True:

        rval, frame = cap.read()

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if rval:
            img_output = algorithm.apply(frame)
            img_bgmodel = algorithm.getBackgroundModel()

            # save_path 設為 f"{vid_name}", save_name 設為 frame_idx
            bgs_frame = lightweight_denoise_update(img_output, debug_mode, save_path, frame_idx, wh_area_threshold)

            bgs_frames.put( bgs_frame, block=True )

        else: break
    cap.release()

def lightweight_denoise_update(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=150):

    # save_path 設為 f"{vid_name}"
    # save_name 設為 frame_idx

    # 輸入為三通道圖片，輸出為單通道的圖片
    # 流程：模糊 -> 二值化 -> 刪除過小區域

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        debug_img = np.zeros((height, width, 3), np.uint8)

    # first process: 模糊、二值化
    mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # second_process: 刪除過小的白色區域
    num_labels, img_with_labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # cc_bboxes 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 異常區域
    start = 0
    while not(cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height):

        start+=1

        if start==cc_bboxes.shape[0]: break

    # 移除 cc_bboxes 黑色區域
    while cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height:

        start+=1

        if start==cc_bboxes.shape[0]: break

    bgs_frame = np.zeros((height, width), np.uint8)
    for component_label in range(start, num_labels):

        if cc_bboxes[component_label][4]>=wh_area_threshold:

            bgs_frame[img_with_labels==component_label] = 255

            if debug_mode:

                debug_img[img_with_labels==component_label] = (255, 255, 255)

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]
                
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

    if debug_mode: 
        cv2.imwrite(f"{save_path}/debug/lightweight_denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(f"{save_path}/debug/lightweight_denoise/second_process/{save_name}.png", debug_img)
    
    # 當有偵測到物體時，會使用到 img_with_labels, valid_labels, cc_bboxes

    # 當沒偵測到物體時，會使用到bgs_frame
    # bgs_frame 為單通道
    # bgs_frame 必須膨脹兩次才符合先前的設定
    # 又因為把denoise_woufg中的一次膨脹移除，所以總共需膨脹3次
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bgs_frame = cv2.dilate(bgs_frame, kernel, iterations=3)

    return bgs_frame


def denoise_woufg(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=600): 

    # save_path 設為 vid_name / debug
    # save_name 設為 frame_idx

    # 輸入為單通道

    try: # input: mask 是二通道的圖片
        height, width = mask.shape
    except: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        debug_img = np.zeros((height, width, 3), np.uint8)

    # first_process: 模糊、二值化、膨脹
    # mask = cv2.medianBlur(mask, 3)
    # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    # mask = cv2.dilate(mask, kernel, iterations=1)

    # second_process: 刪除過小的白色區域
    num_labels, img_with_labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 中異常區域
    remove_idx, start = [], 0 
    while not (cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height):

        remove_idx.append(start)

        start+=1

        if start==cc_bboxes.shape[0]: break
    
    while cc_bboxes[start][0]==0 and cc_bboxes[start][1]==0 and cc_bboxes[start][2]==width and cc_bboxes[start][3]==height:

        remove_idx.append(start)

        start+=1

        if start==cc_bboxes.shape[0]: break

    valid_labels = []
    for component_label in range(start, num_labels):
        
        if cc_bboxes[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果

            valid_labels.append(component_label)
            
            if debug_mode:
                
                debug_img[img_with_labels==component_label] = (255, 255, 255)

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]

                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

        else:
            remove_idx.append(component_label)

    if debug_mode: 
        # first_process
        # cv2.imwrite(f"{save_path}/debug/denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # second_process
        cv2.imwrite(f"{save_path}/debug/denoise/second_process/{save_name}.png", debug_img)
    
    return img_with_labels, set(valid_labels), remove_idx, cc_bboxes



        

def del_cc_region_woufg(args):

    yolo_bbox, component_label, cc_info = args

    yolo_ul, yolo_lr = (int(yolo_bbox[0]), int(yolo_bbox[1])), (int(yolo_bbox[2]), int(yolo_bbox[3]))
    # yolo_area = (yolo_lr[0]-yolo_ul[0])*(yolo_lr[1]-yolo_ul[1]) # w*h

    cc_ul, cc_lr = (cc_info[0], cc_info[1]), (cc_info[0]+cc_info[2], cc_info[1]+cc_info[3])

    intersect_ul, intersect_lr = ( max(yolo_ul[0], cc_ul[0]), max(yolo_ul[1], cc_ul[1]) ), ( min(yolo_lr[0], cc_lr[0]), min(yolo_lr[1], cc_lr[1]) )
    intersect_area = max(intersect_lr[0]-intersect_ul[0], 0)*max(intersect_lr[1]-intersect_ul[1], 0) # w*h

    if not intersect_area == 0.0: 
        return component_label
    return None






















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