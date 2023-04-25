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



def video2frames(path2video, save_path="./result_video2frames", frame_interval=1, save_frames=True, BGR2GRAY=False):
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


def my_denoise(path2img, save_path="./result_my_denoise", save_name="result_my_denoise.png"):

    print("\nProcess: my_denoise")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 灰階讀取
    img = cv2.imread(path2img, cv2.IMREAD_GRAYSCALE)

    # 二值化
    rval, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    '''
        input:
            img = 欲處理圖片
            connectivity 可以選擇4或8連通
                # 4連通代表上下左右
                # 8連通代表上下左右、左上、右上、右下、左下

        output:
            num_labels 為連通域數量
            labels 是一個和img一樣大小的矩形，意即 labels.shape = image.shape ，其中每個連通區域會有一個"唯一"的標誌，從0開始
            
            stats 會包含五個參數 x, y, h, w, s ，分別對應每個連通區域的外接矩形的起始座標 x, y, width, height ， s 為labels對應的連通區域的像素個數。
                # stats = 
                    array(
                       #  x, y, w,  h,  s
                        [[0, 0, 10, 10, 76],  <--- 代表整張圖片
                         [4, 1,  5,  6, 18],  <--- 標記1的區域的資訊
                         [...],               <--- 標記2的區域的資訊
                         ...                  <--- ... 
                         [...]]
                        , dtype=int32
                    )
            
            centroids 中心點
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)

    '''
        stats[0]為整張圖片的資訊，從stats[1]開始，意即扣除整張圖片的部分。
        cv2.CC_STAT_AREA 代表該區域的面積
        areas 為二維陣列，維度 n*1 ， n 為標籤數量，即連通區域數量， area[i] 的內容代表標籤 i 的區域 Pixels 數量
    '''
    areas = stats[:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
    for component_label in range(1, num_labels):            # For 走遍每個連通區域，走遍每種標籤
                                                            # 從 1 開始是因為扣除 labels==0 (二值化圖片黑色區域)

        '''
            labels = np.array([
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 0, 4, 4],
                [3, 3, 0, 4, 4]
             ])

            labels == 0            <--- 回傳一個 np.array ，在 labels 上標籤為 0 的位置會是 True，其他為 False ，這個是二值化圖片的黑色部分，不用檢查
            labels == 1            <--- 回傳一個 np.array ，在 labels 上標籤為 1 的位置會是 True，其他為 False 
                # input: labels == 1
                # output: 
                #   [[T, T, F, F, F],
                #    [T, T, F, F, F],
                #    [F, F, F, F, F],
                #    [F, F, F, F, F],
                #    [F, F, F, F, F]]
            ...
            labels == num_labels-1 <--- 回傳一個 np.array ，在 labels 上標籤為 num_labels-1 的位置會是 True，其他為 False
            
            Eg.
                labels = np.array([
                    [1, 1, 0, 2, 2],
                    [1, 1, 0, 2, 2],
                    [0, 0, 0, 0, 0],
                    [3, 3, 0, 4, 4],
                    [3, 3, 0, 4, 4]
                ])
                result = np.zeros((5, 5), np.uint8)

                for idx in range (3):
                    result[labels==idx] = idx+10
                    print("{}\n".format(result))
        '''

        if areas[component_label] >= 300:                   # 200 應為 pixels 數的 Threshold
            result[labels == component_label] = 255         # 255 為 Color
                                                            # 迴圈會從 1 開始到 num_labels-1 走遍每個標籤
                                                            # 檢查標籤 i 的區域 Pixels 是否多於設定的 Threshold
                                                            # 如果是，則為白色(255)，否則黑色(0)
                                                            # labels == component_label 這條運算式的結果會回傳一個 np.array ，每個 Entry 均為 boolean type

                                                            # result[ labels == component_label ] = 225
                                                            # 左式是一個 2D 的 np.array ，其大小和 labels == component_label 相同
                                                            # 會把 labels == component_label 為 True 的位置改成255， False 的位置為 0
                                                            # 每次 iteration 的結果似乎會與前一次的結果相加

    cv2.imwrite(os.path.join(save_path, save_name), result)


def knn(path2video, save_path="./result_knn", save_name="result_knn.mp4", frame_interval=1):
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
        seg_gray = cv2.imread(os.path.join(seg_algo, seg_frame), cv2.IMREAD_GRAYSCALE)

        # 二值化得到mask
        _, seg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # bitwise operation
        # bg_sub_mask and seg_mask = knn純人像
        # bg_sub_human = cv2.bitwise_and(bg_sub_img, bg_sub_img, mask=seg_mask)

        # # 轉灰度圖，並二值化
        # bg_sub_human_gray = cv2.cvtColor(bg_sub_human, cv2.COLOR_BGR2GRAY)
        # _, bg_sub_human_mask = cv2.threshold(bg_sub_human_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        bg_sub_gray = cv2.cvtColor(bg_sub_img, cv2.COLOR_BGR2GRAY)
        _, bg_sub_mask = cv2.threshold(bg_sub_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # bg_sub_mask xor knn純人像 = 雜訊和垃圾
        trash_and_noise_mask = cv2.bitwise_or(bg_sub_mask, seg_mask)

        # 轉img儲存
        trash_and_noise_img = cv2.cvtColor(trash_and_noise_mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), trash_and_noise_img)

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
    lr_x, lr_y = min(int(bbox[0]+(1.1)*bbox[2]-1), img_width), min(int(bbox[1]+(1.1)*bbox[3]-1), img_height)

    return ul_x, ul_y, lr_x, lr_y


def denoise_on_bin_video(path2video, save_path="./result_denoise_on_bin_video", save_name="result_denoise_on_bin_video.mp4", npy_save_path="./denoise_npy", area_threshold=50, save_bbox=False):

    '''
        每一幀都必須要是GrayScale圖片

        ..._img  為RGB的圖片
        ..._gray 為灰階圖片
        ..._mask 為二值化後的二維圖片
    '''

    # ---------------config--------------- #
    thickness = 2      # 線的厚度
    lineType = 4       # 線的類型
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # ------------------------------ #

    check_video_name(path2video)
    check_video_name(save_name)
    file_name = path2video.split('/')[-1]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
       
    cap = cv2.VideoCapture(path2video)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)  
    else:
        print("Fail to open {}".format(file_name))
        cap.release()
        os._exit(0)
    cap.release()

    frames = video2frames(path2video, frame_interval=1, save_frames=False)
    height, width, channel = frames[0].shape
    size = (width, height)

    videoWrite = cv2.VideoWriter( os.path.join(save_path, save_name), fourcc, fps, size )

    progress = tqdm(total=len(frames))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for idx, frame in enumerate(frames):

        frame = cv2.medianBlur(frame, 3)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
        denoise_img = np.zeros((height, width, 3), np.uint8) 
        for component_label in range(start, num_labels):
            if stats[component_label][4] > area_threshold: # 這個 if 有 Denoise 的效果
                denoise_img[labels==component_label] = (255, 255, 255)
            else:
                remove_idx.append(component_label)

        if save_bbox:
            if not os.path.exists(npy_save_path):
                os.makedirs(npy_save_path)

            interest_bboxes = np.delete(stats, remove_idx, axis=0)
            np.save(f"{npy_save_path}/{idx}.npy", interest_bboxes)

        videoWrite.write(denoise_img)
        
        progress.set_description("[denoise_on_bin_video]")
        progress.update(1)
    videoWrite.release()