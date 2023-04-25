import os
import sys
import time

from tqdm import tqdm, trange
from time import sleep

import re
import glob
from natsort import natsorted
from moviepy.editor import *

import cv2
import numpy as np
import torch

def check_video_name(path2video):
    filename_extension = path2video.split('.')[-1]
    if not filename_extension=="avi" and not filename_extension=="mp4":
        path2video = path2video+".mp4"
    return path2video


def need_modify_video2frames(path2video, save_path="./result_video2frames", frame_interval=1, save_frames=True, BGR2GRAY=False):
    '''
        提取出video中的frames\n
        path2video 為video的路徑，例：./videos/video_1.mp4 \n
        save_path 為儲存frames的資料夾路徑，例：./video_1_frames \n
        frame_interval 為幀的間隔，表示每隔frame_interval取一幀，預設值為1 \n
        save_frames 代表要不要儲存frames，預設值為True，若為False，則會返回np.array \n
        BGR2GRAY 代表要不要把圖片轉成灰階，預設值為False
    '''

    path2video = check_video_name(path2video)
    # read_video 為讀取的video名稱
    read_video = path2video.split('/')[-1]
    
    cap = cv2.VideoCapture(path2video)
    if(cap.isOpened()):
        # 成功讀取影片，檢查save_path
        if save_frames and not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        print("Fail to open {}".format(read_video))
        os._exit(0)

    frame_idx = 0
    save_array = []
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    rval, frame = cap.read()
    while rval:
        if frame_idx%frame_interval == 0:

            # temp
            # frame = np.resize(frame, (960, 540))

            if BGR2GRAY:
                # 輸入為黑白圖片
                save_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                # 輸入為彩圖
                save_array.append(frame)

        progress.set_description("[Extract Frame]")            
        progress.update(1)
        rval, frame = cap.read()
        frame_idx = frame_idx+1
        cv2.waitKey(1)
    cap.release()

    frame_idx = 0
    if save_frames:
        progress = tqdm(total=len(save_array))
        for frame in save_array:
            img_name = "frame_"+str(frame_idx)+".png"
            cv2.imwrite(os.path.join(save_path, img_name),frame)

            frame_idx = frame_idx+1
            progress.set_description("[Save Frame]")            
            progress.update(1)
    else:        
        return save_array


def lightweight_video2frames(path2video, save_path="./result_video2frames", frame_interval=1):
    '''
        提取出video中的frames\n
        path2video 為video的路徑，例：./videos/video_1.mp4 \n
        save_path 為儲存frames的資料夾路徑，例：./video_1_frames \n
        frame_interval 為幀的間隔，表示每隔frame_interval取一幀，預設值為1 \n
        save_frames 代表要不要儲存frames，預設值為True，若為False，則會返回np.array \n
        BGR2GRAY 代表要不要把圖片轉成灰階，預設值為False
    '''

    path2video = check_video_name(path2video)
    # read_video 為讀取的video名稱
    read_video = path2video.split('/')[-1]
    
    cap = cv2.VideoCapture(path2video)
    if(cap.isOpened()):
        # 成功讀取影片，檢查save_path
        if not os.path.exists(save_path): os.makedirs(save_path)
    else:
        print("Fail to open {}".format(read_video))
        os._exit(0)

    frame_idx = 0
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    rval, frame = cap.read()
    while rval:
        
        if frame_idx%frame_interval==0:
            cv2.imwrite(f"{save_path}/{frame_idx}.png", frame)
        
        progress.set_description("[Save Frame]")            
        progress.update(1)
        rval, frame = cap.read()
        frame_idx = frame_idx+1
        cv2.waitKey(1)
    cap.release()


def frames2video(path2frames, save_path="./result_frames2video", save_name="result_frames2video.mp4", get_fps_from=""):    
    '''
        將數張照片串接成video。\n
        path2frame 為儲存frames的資料夾，例：./frames \n
        save_path 為儲存video的資料夾 \n
        save_name 為儲存video的名稱 \n        
        get_fps_from 為video的路徑，例：./videos/video_1.mp4，用於取得的FPS用，預設值為60。
    '''
    save_name = check_video_name(save_name)

    print(
        "\nProcess: frames2video\n"+
        "Connect the frames in: {}\n".format(path2frames)+
        "Save the result to: {}".format(os.path.join(save_path, save_name))
    )

    if get_fps_from=="":
        fps = 60.0
    else:        
        read_video = get_fps_from.split('/')[-1]
        cap = cv2.VideoCapture(get_fps_from)
        # check whether the video can open or not
        if(cap.isOpened()):
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            print("Fail to open {}. Use the default fps=60 to connect each frames.".format(read_video))
            fps = 60.0
        cap.release()

    frames = os.listdir(path2frames)
    frames = natsorted(frames)

    height, width, channel = cv2.imread( os.path.join(path2frames, frames[0]) ).shape

    print(
        "\nThere are {} frames in the folder.\n".format(len(frames))+
        "Some information of the first frames are listed below "+
        " - Height  : {}\n".format(height)+
        " - Width   : {}\n".format(width)+
        " - Channel : {}\n".format(channel)
    )

    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    videoWrite = cv2.VideoWriter( os.path.join(save_path, save_name) , fourcc, fps, size)

    progress = tqdm(total=len(frames))

    for frame in frames:
        img = cv2.imread( os.path.join(path2frames, frame) )
        videoWrite.write(img)
        
        progress.set_description("[frames2video]")            
        progress.update(1)
    videoWrite.release()


def need_modify_knn(path2video, save_path="./result_knn", save_name="result_knn.mp4", frame_interval=1):
    '''
        對video每一幀做knn，並將其儲存成影片 \n
        path2video 為到video的路徑，例：./videos/video.mp4 \n
        save_path 為儲存影片的資料夾，例：./video_1_frames \n
        frame_interval 為幀與幀之間的間隔
    '''

    path2video = check_video_name(path2video)
    save_name = check_video_name(save_name)

    read_video = path2video.split('/')[-1]

    cap = cv2.VideoCapture(path2video)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        print(
            "Fail to open {}.\n".format(read_video)+
            "Please check the path to video again.\n"+
            "The value of function parameter [path2video] is: {}".format(path2video)
        )
        os._exit(0)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 常见一个BackgroundSubtractorKNN接口
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    rval, frame = cap.read()

    height, width, channel = frame.shape

    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(os.path.join(save_path, save_name) , fourcc, fps, size)

    progress = tqdm(total=frame_count)
    frame_idx = 0
    while rval:
        if frame_idx%frame_interval==0:
            # 3. apply()函数计算了前景掩码
            fg_mask = bs.apply(frame)
            # fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            # videoWrite.write(fg_mask)

            # 4. 获得前景掩码（含有白色值以及阴影的灰色值）
            # 通过设定阈值将非白色（244~255）的所有像素都设为0，而不是1
            # 二值化操作
            _, th = cv2.threshold(fg_mask.copy(),244,255,cv2.THRESH_BINARY)
            th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            videoWrite.write(th)

        rval, frame = cap.read()
        frame_idx = frame_idx+1
        progress.set_description("[knn]")
        progress.update(1)
        cv2.waitKey(1)
    cap.release()
    videoWrite.release()


def concat_videos_horizontal(path2videos, save_path="./result_concat_videos_horizontal", save_name="result_concat_videos_horizontal.mp4"):
    
    save_name = check_video_name(save_name)

    videos = os.listdir(path2videos)
    videos = natsorted(videos)
    
    videos_array = []

    print("The order of the video after concat is:")
    for video in videos:

        print(video+" ", end='')

        file = os.path.join(path2videos, video)
        videos_array.append(VideoFileClip(file))
    print("")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output = clips_array([videos_array])
    file = os.path.join(save_path, save_name)
    output.write_videofile(file, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")


def concat_videos_vertical(path2videos, save_path="./result_concat_videos_vertical", save_name="result_concat_videos_vertical.mp4"):

    print("\nProcess: concat_videos_vertical")
    
    save_name = check_video_name(save_name)

    videos = os.listdir(path2videos)
    videos = natsorted(videos)

    videos_array = []

    print("The order of the video after concat is:")
    for video in videos: 

        print(video)

        file = os.path.join(path2videos, video)
        videos_array.append( [VideoFileClip(file)] )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output = clips_array(videos_array)
    file = os.path.join(save_path, save_name)
    output.write_videofile(file, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")


def bitwise_process_xor(bg_sub_algo, seg_algo, save_path):

    # bg_sub_algo 儲存傳統算法輸出mask的資料夾
    # seg_algo 儲存yolov7輸出mask的資料夾

    if not os.path.exists(save_path): os.makedirs(save_path)

    # 讀取檔案
    bg_sub_frames, seg_frames = natsorted(os.listdir(bg_sub_algo)), natsorted(os.listdir(seg_algo))

    progress = tqdm(total=len(bg_sub_frames))
    for idx, (bg_sub_frame, seg_frame) in enumerate(zip(bg_sub_frames, seg_frames)):

        '''
            定義
            ..._gray 為灰階圖片
            ..._mask 為二值化後的圖片
            ..._img  為三個Channel的圖片

            備註
                二值化以後在做bitwise operation物體的輪廓會比較好
        '''

        bg_sub_img = cv2.imread(os.path.join(bg_sub_algo, bg_sub_frame))
        seg_gray = cv2.imread(os.path.join(seg_algo, seg_frame), cv2.IMREAD_GRAYSCALE)

        # 二值化得到mask
        _, seg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # bitwise operation
        # bg_sub_mask and seg_mask = knn純人像
        bg_sub_human = cv2.bitwise_and(bg_sub_img, bg_sub_img, mask=seg_mask)

        # 轉灰度圖，並二值化
        bg_sub_human_gray = cv2.cvtColor(bg_sub_human, cv2.COLOR_BGR2GRAY)
        _, bg_sub_human_mask = cv2.threshold(bg_sub_human_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        bg_sub_gray = cv2.cvtColor(bg_sub_img, cv2.COLOR_BGR2GRAY)
        _, bg_sub_mask = cv2.threshold(bg_sub_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # bg_sub_mask xor knn純人像 = 雜訊和垃圾
        trash_and_noise_mask = cv2.bitwise_xor(bg_sub_mask, bg_sub_human_mask)

        # 轉img儲存
        trash_and_noise_img = cv2.cvtColor(trash_and_noise_mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), trash_and_noise_img)

        progress.update(1)


def bitwise_process_or(bg_sub_algo, seg_algo, save_path):

    # bg_sub_algo 儲存傳統算法輸出mask的資料夾
    # seg_algo 儲存yolov7輸出mask的資料夾

    if not os.path.exists(save_path): os.makedirs(save_path)

    # 讀取檔案
    bg_sub_frames, seg_frames = natsorted(os.listdir(bg_sub_algo)), natsorted(os.listdir(seg_algo))

    progress = tqdm(total=len(bg_sub_frames))
    for idx, (bg_sub_frame, seg_frame) in enumerate(zip(bg_sub_frames, seg_frames)):

        '''
            定義
            ..._gray 為灰階圖片
            ..._mask 為二值化後的圖片
            ..._img  為三個Channel的圖片

            備註
                二值化以後在做bitwise operation物體的輪廓會比較好
        '''

        bg_sub_img = cv2.imread(os.path.join(bg_sub_algo, bg_sub_frame))
        bg_sub_gray = cv2.cvtColor(bg_sub_img, cv2.COLOR_BGR2GRAY)
        _, bg_sub_mask = cv2.threshold(bg_sub_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)


        seg_gray = cv2.imread(os.path.join(seg_algo, seg_frame), cv2.IMREAD_GRAYSCALE)
        # 二值化得到mask
        _, seg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # bg_sub_mask xor knn純人像 = 雜訊和垃圾
        total_mask = cv2.bitwise_or(bg_sub_mask, seg_mask)

        # 轉img儲存
        total_img = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), total_img)

        progress.update(1)


def compute_iou(x_bbox, y_bbox):
    '''
        bbox = 
            array(
                # x, y, w,  h,  s
                [[0, 0, 10, 10, 76],  <--- 代表整張圖片
                 [4, 1,  5,  6, 18],  <--- 標記1的區域的資訊
                 [...],               <--- 標記2的區域的資訊
                  ...                 <--- ... 
                 [...]]
                , dtype=int32
            )    
    '''
    x_ul, x_lr = (x_bbox[0], x_bbox[1]), (x_bbox[0]+x_bbox[2]-1, x_bbox[1]+x_bbox[3]-1)
    y_ul, y_lr = (y_bbox[0], y_bbox[1]), (y_bbox[0]+y_bbox[2]-1, y_bbox[1]+y_bbox[3]-1)

    intersection_ul, intersection_lr = ( max(x_ul[0], y_ul[0]), max(x_ul[1], y_ul[1]) ), ( min(x_lr[0], y_lr[0]), min(x_lr[1], y_lr[1]) ) 

    w, h = max(0, intersection_lr[0]-intersection_ul[0]), max(0, intersection_lr[1]-intersection_ul[1])
    intersection_area = w*h

    if intersection_area==0: return 0
    else: 
        x_area, y_area = x_bbox[2]*x_bbox[3], y_bbox[2]*y_bbox[3]

        union = x_area+y_area-intersection_area
        
        return intersection_area/union


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


def dilate_bbox(bbox, img_height, img_width, dilate_ratio=0.1):
    
    ul_x, ul_y = max(int(bbox[0]-bbox[2]*0.1), 0), max(int(bbox[1]-bbox[3]*0.1), 0)
    lr_x, lr_y = min(int(bbox[0]+(1.1)*bbox[2]), img_width), min(int(bbox[1]+(1.1)*bbox[3]), img_height)

    return (ul_x, ul_y), (lr_x, lr_y)


def dilate_bbox_beta(bbox, img_height, img_width, dilate_ratio=0.1):
    
    ul_x, ul_y = max(int(bbox[0]-bbox[2]*0.1), 0), max(int(bbox[1]-bbox[3]*0.1), 0)
    lr_x, lr_y = min(int(bbox[0]+(1.1)*bbox[2]-1), img_width), min(int(bbox[1]+(1.1)*bbox[3]-1), img_height)

    return ul_x, ul_y, lr_x, lr_y


def intersection(x_bbox, y_bbox):
    
    x_ul, x_lr = (x_bbox[0], x_bbox[1]), (x_bbox[0]+x_bbox[2]-1, x_bbox[1]+x_bbox[3]-1)
    y_ul, y_lr = (y_bbox[0], y_bbox[1]), (y_bbox[0]+y_bbox[2]-1, y_bbox[1]+y_bbox[3]-1)

    intersection_ul, intersection_lr = ( max(x_ul[0], y_ul[0]), max(x_ul[1], y_ul[1]) ), ( min(x_lr[0], y_lr[0]), min(x_lr[1], y_lr[1]) ) 

    w, h = max(0, intersection_lr[0]-intersection_ul[0]), max(0, intersection_lr[1]-intersection_ul[1])
    intersection_area = w*h

    return intersection_area


def get_interest_bbox(mask, wh_area_threshold=800):

    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 設定kernel、寬、高
    height, width = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask = cv2.medianBlur(mask, 3)

    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # stats 記錄了所有連通白色區域的 BBoxes 訊息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 移除 stats 中黑色區域
    remove_idx, start = [], 0
    tmp = stats[start]
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==width and tmp[3]==height:

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]

    # initialize denoise_mask
    denoise_mask = np.zeros((height, width, 3), np.uint8) 
    for component_label in range(start, num_labels):

        if stats[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            denoise_mask[labels==component_label] = (255, 255, 255)

        else:
            remove_idx.append(component_label)

    interest_bboxes = np.delete(stats, remove_idx, axis=0)

    return denoise_mask, interest_bboxes


def denoise(mask, wh_area_threshold=800):
    # input: mask 是三通道的圖片

    # 設定高、寬
    height, width, _ = mask.shape
    
    # 模糊、灰階、二值化
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 膨脹
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=3)

    # 連通域處理
    # stats 記錄了所有連通白色區域的 BBoxes 訊息
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    denoise_mask = np.zeros((height, width, 3), np.uint8) # initialize denoise_mask

    for component_label in range(1, num_labels):

        if stats[component_label][4]>=wh_area_threshold: # 這個 if 有 Denoise 的效果
            denoise_mask[labels==component_label] = (255, 255, 255)

    # 回傳三通道的Denoise Mask
    return denoise_mask


def img_processing(rgb_frame, seg_mask, bgs_mask, resize_width, resize_height, area_ratio=0.5):

    # input: rgb_frame, seg_mask, bgs_mask均為3 Channel

    # Resize(width, height)

    # resize RGB_Frame
    rgb_frame = cv2.resize(rgb_frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    # resize, 灰階, 二值化
    seg_mask = cv2.resize(seg_mask, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    _, seg_mask = cv2.threshold(seg_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 降躁, resize, 灰階, 二值化
    bgs_mask = denoise(bgs_mask)
    bgs_mask = cv2.resize(bgs_mask, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    bgs_mask = cv2.cvtColor(bgs_mask, cv2.COLOR_BGR2GRAY)
    _, bgs_mask = cv2.threshold(bgs_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 不論前景或背景都會在total mask中呈現
    total_mask = cv2.bitwise_or(seg_mask, bgs_mask)

    _, _, stats, _ = cv2.connectedComponentsWithStats(total_mask, connectivity=8)

    # 移除 stats 中黑色區域
    remove_idx, start = [], 0
    tmp = stats[start]
    while tmp[0]==0 and tmp[1]==0 and tmp[2]==resize_width and tmp[3]==resize_height:

        remove_idx.append(start)

        start+=1 # 下一個idx

        # 保護機制
        if start==stats.shape[0]:
            break

        tmp = stats[start]
    interest_bboxes = np.delete(stats, remove_idx, axis=0)

    # 比較total mask和bgs_mask
    # 檢查每個物件占前景的比例為何，過低者刪除
    for bbox in interest_bboxes:

        # 切割total_mask的影像，設為x
        # 以相同BBOX切割bgs_mask，設為y

        bbox_area = bbox[2]*bbox[3]

        ul, lr = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        x, y = total_mask[ul[1]:lr[1], ul[0]:lr[0]], bgs_mask[ul[1]:lr[1], ul[0]:lr[0]]

        # total_mask(fg+bg), bgs_mask(fg)相同區域相減以後，值為1的部分如果超過"總面積*area_ratio"那就代表該區塊為bg，該區域占前景的比例少，要刪除。
        # 反之如果值為1的部分小於"總面積*area_ratio"，則占前景的比例大，要保留。

        if np.count_nonzero(x-y)>=bbox_area*area_ratio:
            total_mask[ul[1]:lr[1], ul[0]:lr[0]] = np.zeros((bbox[3], bbox[2]))

    # RGB_Frame為三通道、Total_Mask為二通道
    return rgb_frame, total_mask