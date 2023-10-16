import os
from tqdm import tqdm
from natsort import natsorted

import cv2
import pybgs as bgs
import numpy as np

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

def bgs_generator(path2rgb):

    cap = cv2.VideoCapture(path2rgb)
    if not cap.isOpened():
        print(f"Fail to open {path2rgb}")
        os._exit(0)

    algorithm = bgs.ViBe()

    pos_frame = cap.get(1)
    while True:

        rval, frame = cap.read()

        if rval:
            pos_frame = cap.get(1)      
            img_output = algorithm.apply(frame)
            img_bgmodel = algorithm.getBackgroundModel()
            yield img_output

        else: break
    cap.release()

def bgs_generator_for_mp(path2rgb, queue):

    cap = cv2.VideoCapture(path2rgb)
    if not cap.isOpened():
        print(f"Fail to open {path2rgb}")
        os._exit(0)

    algorithm = bgs.ViBe()

    pos_frame = cap.get(1)
    while True:

        rval, frame = cap.read()

        if rval:
            pos_frame = cap.get(1)      
            img_output = algorithm.apply(frame)
            img_bgmodel = algorithm.getBackgroundModel()
            queue.put(img_output, block=True)
        else: break
    cap.release()

def denoise(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=600): 

    # save_path 設為 vid_name / debug
    # save_name 設為 frame_idx

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        if not os.path.exists(save_path+"/denoise/first_process"): os.makedirs(save_path+"/denoise/first_process")
        if not os.path.exists(save_path+"/denoise/second_process"): os.makedirs(save_path+"/denoise/second_process")

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape # 設定高、寬
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape # 設定高、寬

    # first_process: 模糊、二值化、膨脹
    mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    mask = cv2.dilate(mask, kernel, iterations=1)

    if debug_mode: cv2.imwrite(f"{save_path}/denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # 連通域處理
    num_labels, labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 中異常區域
    remove_idx, start = [], 0
    tmp = cc_bboxes[start]    
    while not (tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height):

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==cc_bboxes.shape[0]: break

        tmp = cc_bboxes[start]
    
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height:

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==cc_bboxes.shape[0]: break

        tmp = cc_bboxes[start]

    # second_process: 刪除過小的白色區域

    denoise_mask = np.zeros((height, width, 3), np.uint8) # initialize denoise_mask
    if debug_mode: debug_img = denoise_mask.copy()

    for component_label in range(start, num_labels):

        if cc_bboxes[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            denoise_mask[labels==component_label] = (255, 255, 255)
            
            if debug_mode:
                
                debug_img[labels==component_label] = (255, 255, 255)

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]

                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

        else:
            remove_idx.append(component_label)


    if debug_mode: cv2.imwrite(f"{save_path}/denoise/second_process/{save_name}.png", debug_img)
    
    cc_bboxes = np.delete(cc_bboxes, remove_idx, axis=0)
    # 回傳三通道的Denoise Mask
    return denoise_mask, cc_bboxes
    
def denoise_if_detect_obj(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=600):

    # save_path 設為 vid_name / debug
    # save_name 設為 frame_idx

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        if not os.path.exists(save_path+"/denoise/first_process"): os.makedirs(save_path+"/denoise/first_process")
        if not os.path.exists(save_path+"/denoise/second_process_del"): os.makedirs(save_path+"/denoise/second_process_del")

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape # 設定高、寬
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape # 設定高、寬

    # first_process: 模糊、二值化、膨脹
    mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    mask = cv2.dilate(mask, kernel, iterations=1)

    if debug_mode: cv2.imwrite(f"{save_path}/denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # 連通域處理
    num_labels, labels, cc_bboxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 cc_bboxes 中異常區域
    start = 0
    remove_labels = []
    tmp = cc_bboxes[start]    
    while not (tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height):

        remove_labels.append(start)

        start+=1 # 下一個idx

        if start==cc_bboxes.shape[0]: break

        tmp = cc_bboxes[start]
    
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height:

        remove_labels.append(start)

        start+=1 # 下一個idx

        if start==cc_bboxes.shape[0]: break

        tmp = cc_bboxes[start]

    # second_process: 刪除過小的白色區域
    if debug_mode: debug_img = np.zeros((height, width, 3), np.uint8)

    reserve_labels = []
    denoise_mask = np.zeros((height, width, 3), np.uint8) # initialize denoise_mask
    for component_label in range(start, num_labels):

        if cc_bboxes[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            denoise_mask[labels==component_label] = (255, 255, 255)
            reserve_labels.append(component_label)
        else:
            remove_labels.append(component_label)

            if debug_mode:

                debug_bbox = cc_bboxes[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]
                
                debug_img[labels==component_label] = (255, 255, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

    if debug_mode: cv2.imwrite(f"{save_path}/denoise/second_process_del/{save_name}.png", debug_img)
         
    # 回傳內容如下
    # 三通道的Denoise Mask: denoise_mask
    # 帶有label的二維陣列: labels
    # 二次降躁前，所有連通域的bboxes: cc_bboxes
    # 被保留的連通域清單: reserve_labels
    # 後續將被刪除的連通域清單: remove_labels
    return denoise_mask, labels, cc_bboxes, reserve_labels, remove_labels

def lightweight_denoise_for_mp(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=150):

    # save_path 設為 f"{vid_name}/debug"
    # save_name 設為 frame_idx

    # 輸入為三通道的灰階圖片，輸出為單通道的圖片
    # 流程：降躁 -> 灰階 -> 膨脹

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        if not os.path.exists(save_path+"/debug/lightweight_denoise/first_process"): os.makedirs(save_path+"/debug/lightweight_denoise/first_process")
        if not os.path.exists(save_path+"/debug/lightweight_denoise/second_process_del"): os.makedirs(save_path+"/debug/lightweight_denoise/second_process_del")

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape

    # first process: 模糊、二值化
    mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if debug_mode: cv2.imwrite(f"{save_path}/debug/lightweight_denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # 連通域處理
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 stats 異常區域
    remove_idx, start = [], 0
    tmp = stats[start]
    while not(tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height):

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]

    # 移除 stats 黑色區域
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height:

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]

    # second_process: 刪除過小的白色區域
    if debug_mode: debug_img = np.zeros((height, width, 3), np.uint8)

    denoise_mask = np.zeros((height, width, 3), np.uint8) # initialize denoise_mask
    for component_label in range(start, num_labels):

        if stats[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            denoise_mask[labels==component_label] = (255, 255, 255)
        
        else:
            if debug_mode:
                debug_bbox = stats[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]
                
                debug_img[labels==component_label] = (255, 255, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

    if debug_mode: cv2.imwrite(f"{save_path}/debug/lightweight_denoise/second_process_del/{save_name}.png", debug_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    denoise_mask = cv2.cvtColor(denoise_mask, cv2.COLOR_BGR2GRAY)
    denoise_mask = cv2.dilate(denoise_mask, kernel, iterations=2)
    
    return denoise_mask

def lightweight_denoise_for_mp_update(mask, debug_mode=False, save_path=None, save_name=None, wh_area_threshold=150):

    # save_path 設為 f"{vid_name}/debug"
    # save_name 設為 frame_idx

    # 輸入為三通道圖片，輸出為單通道的圖片
    # 流程：模糊 -> 二值化 -> 刪除過小區域 -> 灰階 -> 膨脹

    if debug_mode:
        assert not save_path==None
        assert not save_name==None

        color = (0, 0, 255)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        if not os.path.exists(save_path+"/debug/lightweight_denoise/first_process"): os.makedirs(save_path+"/debug/lightweight_denoise/first_process")
        if not os.path.exists(save_path+"/debug/lightweight_denoise/second_process"): os.makedirs(save_path+"/debug/lightweight_denoise/second_process")

    try: # input: mask 是三通道的圖片
        height, width, _ = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except: # input: mask 是二通道的圖片
        height, width = mask.shape

    # first process: 模糊、二值化
    mask = cv2.medianBlur(mask, 3)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if debug_mode: cv2.imwrite(f"{save_path}/debug/lightweight_denoise/first_process/{save_name}.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # 連通域處理
    num_labels, img_with_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # stats 記錄了所有連通白色區域的 BBoxes 訊息

    # 移除 stats 異常區域
    start = 0
    tmp = stats[start]
    while not(tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height):

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]

    # 移除 stats 黑色區域
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height:

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]

    # second_process: 刪除過小的白色區域
    if debug_mode: debug_img = np.zeros((height, width, 3), np.uint8)

    valid_labels = []
    for component_label in range(start, num_labels):

        if stats[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果

            valid_labels.append(component_label)

            if debug_mode:
                debug_bbox = stats[component_label]
                x, y, w, h = debug_bbox[0], debug_bbox[1], debug_bbox[2], debug_bbox[3]

                debug_img[img_with_labels==component_label] = (255, 255, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(debug_img, f"{debug_bbox[4]}", (x, y - 10), font, font_scale, color, font_thickness)

    if debug_mode: cv2.imwrite(f"{save_path}/debug/lightweight_denoise/second_process/{save_name}.png", debug_img)
    
    return img_with_labels, valid_labels, stats

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