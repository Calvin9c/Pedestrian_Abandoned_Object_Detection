import os

import cv2
import numpy as np

from tqdm import tqdm

from natsort import natsorted
from moviepy.editor import *

def check_video_name(path2video):
    filename_extension = path2video.split('.')[-1]
    if not filename_extension=="avi" and not filename_extension=="mp4":
        path2video = path2video+".mp4"
    return path2video


def check_path_exists(path):
    if not os.path.exists(path):
        print(f"{path} are not exist, please check the function variable again.")
        return False
    return True

def video2frames(path2video, save_path="./result_video2frames"):
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        print("Fail to open {}".format(read_video))
        os._exit(0)

    frame_idx = 0
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    rval, frame = cap.read()
    while rval:

        cv2.imwrite(f"{save_path}/{frame_idx}.png", frame)

        progress.set_description("[Extract Frame]")            
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

def compute_roi(x_bbox, y_bbox):
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

def mask_video(path2video, path2mask, save_path="./result_mask_video", save_name="result_mask_video.mp4"):

    path2video = check_video_name(path2video)
    path2mask = check_video_name(path2mask)
    save_name = check_video_name(save_name)

    frames = video2frames(path2video, save_frames=False)
    masks = video2frames(path2mask, save_frames=False, BGR2GRAY=True)

    read_video = path2video.split('/')[-1]
    cap = cv2.VideoCapture(path2video)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        print(">> Fail to open {}.".format(read_video))
        os._exit(0)
    cap.release()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    height, width, channel = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter( os.path.join(save_path, save_name), fourcc, fps, size)

    progress = tqdm(total=len(frames))
    masks_idx = 0
    for frame in frames:

        rval, mask = cv2.threshold(masks[masks_idx], 127, 255, cv2.THRESH_BINARY)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        videoWriter.write(result)

        masks_idx = masks_idx+1
        progress.set_description("[mask_video]")
        progress.update(1)
    videoWriter.release()

def extract_fg_by_mask(path2bg, path2fg, save_path="./result_extract_fg_by_mask", save_name="result_extract_fg_by_mask.png"):

    '''
        使用mask提取原圖特定區域
    '''

    bg = cv2.imread(path2bg)
    fg = cv2.imread(path2fg)

    # Perform image substraction
    diff = cv2.absdiff(bg, fg)

    # Convert the result to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the result to obtain a binary mask
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # save the result
    cv2.imwrite('./fg_objs.png', thresh)

def lightweight_video2frames(path2video, save_path="./result_video2frames", need_resize=False, resize_height=540, resize_width=960, frame_interval=1):
    '''
        提取出video中的frames\n
        path2video 為video的路徑，例：./videos/video_1.mp4 \n
        save_path 為儲存frames的資料夾路徑，例：./video_1_frames \n
        frame_interval 為幀的間隔，表示每隔frame_interval取一幀，預設值為1 \n
        save_frames 代表要不要儲存frames，預設值為True，若為False，則會返回np.array \n
        BGR2GRAY 代表要不要把圖片轉成灰階，預設值為False
    '''

    if need_resize:
        assert not resize_height==0 and not resize_width==0, "Please check the value of resize_height and resize_width."
    
    path2video = check_video_name(path2video)
    read_video = path2video.split('/')[-1] # read_video 為讀取的video名稱
    
    cap = cv2.VideoCapture(path2video)
    if(cap.isOpened()):
        # 成功讀取影片，檢查save_path
        if not os.path.exists(save_path): os.makedirs(save_path)

        frame_height, frame_width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if resize_height==frame_height and resize_width==frame_width: need_resize=False

    else:
        print("Fail to open {}".format(read_video))
        os._exit(0)

    frame_idx = 0
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    rval, frame = cap.read()
    while rval:
        if frame_idx%frame_interval==0:

            if need_resize: frame = cv2.resize(frame, (resize_width, resize_height))

            cv2.imwrite(f"{save_path}/{frame_idx}.png", frame)
        
        progress.set_description("[Save Frame]")            
        progress.update(1)
        rval, frame = cap.read()
        frame_idx = frame_idx+1
        cv2.waitKey(1)
    cap.release()

def modify_video(path2video, save_name=None, save_path="./result_modify_video", resize=False, resize_height=-1, resize_width=-1, frame_interval=1):

    if not os.path.exists(path2video):
        print( "Fail to open "+path2video )
        os._exit(0)

    cap = cv2.VideoCapture(path2video)
    
    if save_name==None:
        save_name = path2video.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if resize:
        assert not (resize_height==-1 and resize_width==-1), "Please set up resize_height, resize_width."

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if resize_height==-1 else resize_height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if resize_width==-1 else resize_width

    else:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    modified_vid = cv2.VideoWriter(f"{save_path}/{save_name}", fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=frame_count)
    for frame_idx in range(frame_count):

        rval, frame = cap.read()
        if not rval:
            print(f"Fail to read frame {frame_idx}.\nKill the process.")
            cap.release()
            os._exit(0)
        
        if not frame_idx%frame_interval==0: 
            progress.update(1)
            continue

        if resize:
            frame = cv2.resize(frame, (width, height))
        
        modified_vid.write(frame)
        progress.update(1)
    
    cap.release()
    modified_vid.release()