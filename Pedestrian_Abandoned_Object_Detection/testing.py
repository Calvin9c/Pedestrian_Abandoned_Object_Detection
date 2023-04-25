from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from PIL import Image
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn

from evaluate_gpu_ver2 import evaluate
import cv2

from natsort import natsorted
import yh as yh
from tqdm import tqdm, trange


def get_score():
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()
    # print(query_feature.shape)
    for i in range(1):
        score = evaluate(query_feature[i],gallery_feature)
    return score[0]


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def load_network(network):
    save_path = os.path.join('./model',"ft_ResNet50",'net_59.pth')
    network.load_state_dict(torch.load(save_path))
    return network


def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    # print(dataloaders.shape)
    count = 0

    for iter, data in enumerate(dataloaders):
        # img, label = data
        img = data
        
        img = img.unsqueeze(0)
        n, c, h, w = img.size()
        count += n
        # print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        ms = [1]

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor( 1, ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*32
        # end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        end = min( (iter+1)*32, 1)
        features[ start:end, :] = ff
    return features


def load_model():    
    config_path = os.path.join('./model',"ft_ResNet50",'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    model_structure = ft_net(751, stride = 2, ibn = "false", linear_num=512)
    model = load_network(model_structure)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.cuda()
    model = fuse_all_conv_bn(model)

    h, w = 256, 128
    dummy_forward_input = torch.rand(32, 3, h, w).cuda()
    model = torch.jit.trace(model, dummy_forward_input)
    return model


def similarity(array1,array2,model):
    with torch.no_grad():
        gallery_feature = extract_feature(model,array1)
        query_feature = extract_feature(model,array2)
    result = {'gallery_f':gallery_feature.numpy(),'query_f':query_feature.numpy()}
    scipy.io.savemat('pytorch_result.mat',result)
    score = get_score()
    return score


def check_bbox(u_bbox, v_bbox):
    # x, y, w, h
    ux, uy, uw, uh = u_bbox[0], u_bbox[1], u_bbox[2], u_bbox[3]
    vx, vy, vw, vh = v_bbox[0], v_bbox[1], v_bbox[2], v_bbox[3]

    # 定義長寬差距
    wh_threshold = 10 # pixel

    # 1. 長寬差太多
    if abs(uw-vw)>10 or abs(uh-vh)>10:
        return False # 兩者不同物
    else:
        return True

    # 2. Box 間距離太遠


def checking(box1, box2, overlap_th):  # 檢查若 1.長寬差兩倍以上 2.重疊程度在overlap_th以下，返回false 即代表不是同個物件
    check = 1
    # 若長or寬兩倍以上，不會認定是同物體
    if box1[2]/box2[2] > 2 or box2[2]/box1[2] > 2 or box1[3]/box2[3] > 2 or box2[3]/box1[3] > 2:
        check = 0
    # 若iou小於閥值（重疊太少），不是同物品
    if yh.iou(box1, box2) < overlap_th:
        check = 0

    return check


def test():

    model = load_model()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    a = cv2.imread("./394.png")
    b = cv2.imread("./395.png")

    tensor_y_cropped_img, tensor_x_cropped_img = transform(a).unsqueeze(0), transform(b).unsqueeze(0)

    print(f"sim: {similarity(tensor_y_cropped_img, tensor_x_cropped_img, model)}")





def main_copy():
    '''
        ./our_datum
            ├ input                         <----- path2input
            │   ├ video_1.mp4               <----- rgb_cap
            │   ├ ...
            │   └ video_n.mp4
            ├ tmp
            │   ├ yolo_seg                  <----- path2seg_folders
            │   │   ├ video_1               <----- path2seg_masks
            │   │   │   ├ frame_0.png       <----- seg_masks = ["frame_0.png", ..., "frame_m.png"]
            │   │   │   ├ ...
            │   │   │   └ frame_m.png
            │   │   │
            │   │   ├ ...
            │   │   │   
            │   │   └ video_n
            │   │       ├ frame_0.png
            │   │       ├ ...
            │   │       └ frame_m.png
            │   │   
            │   └ bgs_videos                <----- path2bgs_videos
            │       ├ vibe_1.mp4            <----- bgs_cap
            │       ├ ...
            │       └ vibe_n.mp4
            │
            └ output
                ├ video_1                   <----- path2output
                │   ├ rgb
                │   │   ├ frame_0.png
                │   │   ├ ...
                │   │   └ frame_m.png
                │   │   
                │   └ mask
                │       ├ frame_0.png
                │       ├ ...
                │       └ frame_m.png
                │
                ├ ...
                │
                └ video_n

    '''


    path2input, path2seg_folders, path2bgs_videos = "./our_datum/input", "./our_datum/tmp/yolo_seg", "./our_datum/tmp/bgs_videos"

    if not os.path.exists(path2input): 
        print(f"Fail to get any input. Kill the process.")
        os._exit(0)

    if not os.path.exists(path2seg_folders):
        print(f"Fail to get seg_folder. Kill the process.")
        os._exit(0)

    if not os.path.exists(path2bgs_videos):
        print(f"Fail to get bgs_videos. Kill the process.")
        os._exit(0)

    rgb_videos, bgs_videos, seg_mask_folders = natsorted(os.listdir(path2input)), natsorted(os.listdir(path2bgs_videos)), natsorted(os.listdir(path2seg_folders))

    # 建立模型
    model = load_model()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Set the Thresholds
    iou_th = 0.5
    similarity_th = 0.9
    duration_th = 10
    vanish_th = 5
    stay_up_th = 250    # 存在超過此幀數，即使在前景消失也不會被消除
    overlap_th = 0.3    # iou沒達到此閥值不會是同物體
    delete_cnt = 5      # 連續幾幀沒被配到前景就會被刪掉

    # Draw bbox
    # 線的厚度, 類型
    thickness, lineType, font = 2, 4, cv2.FONT_HERSHEY_SIMPLEX

    # 影片每幀Resize長寬
    resize_width, resize_height = 960, 540

    '''
        設定影片播放範圍
        videos_range[0] = (500, 700) 代表處理第一部影片的500到700幀
        videos_range[1] = (0, 0)     代表處理第二部影片的所有幀
        videos_range[2] = (1, 10)    代表處理第二部影片的所有幀
        如果有第四部影片，卻只設定三部影片的範圍，會對第四部影片的每一幀都做處理
    '''
    videos_need_process = [0]
    videos_range = [(390, 400)] # 此處設定影片播放範圍處，從第0幀開始算

    need_append = len(rgb_videos) - len(videos_range)
    if need_append<0:
        print(f"The playback time of the video is set incorrectly. Kill the proces.")
        os._exit(0)
    elif need_append>0:
        for idx in range(need_append): videos_range.append((0, 0))

    # 開始處理
    for video_idx, video in enumerate(rgb_videos):

        # 設定哪幾部影片要處理
        if not video_idx in videos_need_process: continue

        # 設定輸出資料夾
        path2output = f"./our_datum/output/{video_idx}"
        if not os.path.exists(f"{path2output}/mask"): os.makedirs(f"{path2output}/mask") 
        if not os.path.exists(f"{path2output}/rgb"): os.makedirs(f"{path2output}/rgb")

        # 讀取影片
        current_rgb_video, current_bgs_video = f"{path2input}/{video}", f"{path2bgs_videos}/{bgs_videos[video_idx]}"
        rgb_cap = cv2.VideoCapture(current_rgb_video) # 彩色影片，讀取彩幀用
        bgs_cap = cv2.VideoCapture(current_bgs_video) # Bgs影片

        # 檢查是否成功打開影片
        if not rgb_cap.isOpened(): 
            print(f"Fail to open: {current_rgb_video}. Kill the process.")
            os._exit(0)
        if not bgs_cap.isOpened(): 
            print(f"Fail to open: {current_bgs_video}. Kill the process.")
            os._exit(0)

        # 取得幀數, 影片長寬
        total_frames = int( rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT) )
        # width, height = int( rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH) ), int( rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

        # 設定yolo_seg路徑
        path2seg_masks = f"{path2seg_folders}/{seg_mask_folders[video_idx]}"
        seg_masks = natsorted(os.listdir(path2seg_masks))  

        # 初始化上一幀
        previous_rgb_frame = None

        # Initialize Tracking List
        tracking_list = []

        # 設定影片範圍
        video_start, video_end = videos_range[video_idx][0], videos_range[video_idx][1]        
        if video_start==0 and video_end==0:
            video_end = total_frames-1
        elif video_start<0 or video_end>total_frames-1 or video_end-video_start<0 :
            print(f"video_start of {video} is set incorrectly. Kill the process.")
            os._exit(0)
        
        print(f"Processing video: {video_idx}")

        for current_frame_idx in range(total_frames):

            rgb_rval, rgb_frame = rgb_cap.read() # 拍攝的彩色圖片
            bgs_rval, bgs_mask = bgs_cap.read() # Bgs的黑白圖片

            # 如果讀取失敗，程式終止
            if not rgb_rval: 
                print(f"Fail to read the {current_frame_idx} frame of {current_rgb_video}. Kill the process.")
                os._exit(0)
            if not bgs_rval: 
                print(f"Fail to read the {current_frame_idx} frame of {current_bgs_video}. Kill the process.")
                os._exit(0)

            # 播放範圍
            if current_frame_idx<video_start: continue
            if current_frame_idx>video_end: break

            seg_mask = cv2.imread(f"{path2seg_masks}/{seg_masks[current_frame_idx]}") # Seg的黑白圖片

            # 每一幀Resize、刪除占前景比例少的部分在yh.img_processing中實作
            rgb_frame, total_mask = yh.img_processing(rgb_frame, seg_mask, bgs_mask, resize_width, resize_height)

            # 刪除過小的BBOX的部分在yh.get_interest_bbox中實作
            denoise_mask, interest_bboxes = yh.get_interest_bbox(total_mask, wh_area_threshold=800)

            wait_to_add = [] # wait_to_add 用來儲存加入清單之物體

            # wait to merge...

            # Tracking_Code (Before)
            # 設 x 為 tracking_list 中的物體; y 為新一幀的物體
            for y_idx, y_bbox in enumerate(interest_bboxes):

                find_y_in_list = False


                # 放大BBOX計算相似度w
                ul, lr = yh.dilate_bbox(y_bbox, resize_height, resize_width)
                y_cropped_img = rgb_frame[ul[1]:lr[1], ul[0]:lr[0]]
                cv2.imwrite(f"./crop/f_{current_frame_idx}_y_{y_idx}.png", y_cropped_img)

                
                # Handle Case 1 and 2 in our algorithm
                # 用新一幀的物體對Tracking List物體做比對
                # 此處 Tracking List Size 只增不減
                for x_idx, x in enumerate(tracking_list): 
                    # 須考慮 Tracking List 為空的狀況; 
                    # 如果空，不會進for loop

                    x_area, y_area, intersection_area = x["bbox"][2]*x["bbox"][3], y_bbox[2]*y_bbox[3], yh.intersection(x["bbox"], y_bbox)
                    if x_area==intersection_area: # x落在y內
                        find_y_in_list = True
                        break

                    if y_area==intersection_area: # y落在x內
                        find_y_in_list = True
                        break

                    '''
                        # 以下註解可改
                        # 同時，以下註解內容應等價於上述
                        y_ul, y_lr = (y_bbox[0], y_bbox[1]), (y_bbox[0]+y_bbox[2]-1, y_bbox[1]+y_bbox[3]-1)
                        x_ul, x_lr = (x["bbox"][0], x["bbox"][1]), (x["bbox"][0]+x["bbox"][2]-1, x["bbox"][1]+x["bbox"][3]-1)
                    '''

                    # Jamie, Lj
                    # y_ul_x, y_ul_y, y_lr_x, y_lr_y = y_bbox[0], y_bbox[1], y_bbox[0] + y_bbox[2], y_bbox[1] + y_bbox[3]
                    # x_ul_x, x_ul_y, x_lr_x, x_lr_y = x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]
                    # if y_ul_x > x_ul_x and y_ul_y > x_ul_y and y_lr_x < x_lr_x and y_lr_y < x_lr_y:
                    #     find_y_in_list = True
                    #     break

                    # 如果x落在y裡面就不用更新
                    # if x_ul_x > y_ul_x and x_ul_y > y_ul_y and x_lr_x < y_lr_x and x_lr_y < y_lr_y:
                    #     find_y_in_list = True
                    #     break

                    # 放大BBOX計算相似度w
                    # ul, lr = yh.dilate_bbox(y_bbox, resize_height, resize_width)
                    # y_cropped_img = rgb_frame[ul[1]:lr[1], ul[0]:lr[0]]
                    # cv2.imwrite(f"f_{current_frame_idx}_y_{y_idx}.png", y_cropped_img)

                    ul, lr = yh.dilate_bbox(x["bbox"], resize_height, resize_width)
                    x_cropped_img = previous_rgb_frame[ul[1]:lr[1], ul[0]:lr[0]]
                    cv2.imwrite(f"./crop/f_{current_frame_idx}_x_{x_idx}.png", x_cropped_img)

                    tensor_y_cropped_img, tensor_x_cropped_img = transform(y_cropped_img).unsqueeze(0), transform(x_cropped_img).unsqueeze(0)

                    tmp_similarity = similarity(tensor_y_cropped_img, tensor_x_cropped_img, model)
                    print(f"f_{current_frame_idx}: sim of y_{y_idx} and x_{x_idx} is {tmp_similarity}")

                    # 檢查大小 iou有沒有過門檻
                    tmp_checking = checking(y_bbox, x["bbox"], overlap_th)

                    # Case 1: 已追蹤之物體
                    if tmp_similarity>=similarity_th and tmp_checking and x["ref"]==False: # 新一幀物體 y 與清單物體 x 匹配成功
                    
                        if x["state"] == -1: # 匹配到遺失物
                            x["bbox"] = y_bbox
                            x["ref"] = True
                            x["exist"] = True
                            x["duration"] = x["duration"]+1
                            x["vanish_counter"] = 0
                            x["isMoving"] = False
                            x["state"] = 1
                            x["no_pair_counter"] = 0

                        elif x["state"] == 1: # 追蹤中
                            # 更新物件資訊
                            x["bbox"] = y_bbox
                            x["ref"] = True
                            x["exist"] = True
                            x["duration"] = x["duration"]+1
                            x["vanish_counter"] = 0
                            x["isMoving"] = False
                            x["state"] = 1
                            x["no_pair_counter"] = 0
                        
                        if x["duration"] > stay_up_th:
                            x["is_garbage"] = True
                            

                        # 成功在清單中找到，結束比較
                        find_y_in_list = True
                        break

                # 沒在清單中找到物體，此為 Case 2: y 為新進畫面之物件
                if not find_y_in_list:

                    tmp = {
                        "bbox": y_bbox,
                        "frame_idx": current_frame_idx,
                        "ref": True,
                        "exist": True,
                        "duration": 1,
                        "vanish_counter": 0,
                        "isMoving": False,
                        "state": 0,
                        "no_pair_counter": 0,
                        "is_garbage": False,
                    }

                    wait_to_add.append(tmp)

            # 新一幀畫面的所有物體 / 有被框到的所有物體 已處理完畢
    
            # 先處理 Tracking List 中沒出現在畫面中的內容物，即 Case 3
            # 最後再新增新畫面物體到Tracking List
            # 這是為了減少檢查的複雜度

            # Case 3
            # 未出現在新幀畫面之物體
            # 先更新每個obj的參數
            remove_idx = []
            for z_idx, z in enumerate(tracking_list):
                
                # 有被參考，且在Tracking List中，跳下一個物體
                if z["ref"] == True: continue
                
                z["no_pair_counter"] += 1
                if z["no_pair_counter"] >= delete_cnt and not z["is_garbage"]:  # 連續delete_cnt幀沒配到前景，且非垃圾，刪除
                    remove_idx.append(z_idx)
                    continue
                
                # 沒出現的 z ，進入以下邏輯判斷
                z_ul, z_lr = yh.dilate_bbox(z["bbox"], resize_height, resize_width)
                z_current, z_previous = rgb_frame[z_ul[1]:z_lr[1], z_ul[0]:z_lr[0]], previous_rgb_frame[z_ul[1]:z_lr[1], z_ul[0]:z_lr[0]]
                tensor_z_current, tensor_z_previous = transform(z_current).unsqueeze(0), transform(z_previous).unsqueeze(0)

                similarity_val = similarity(tensor_z_current, tensor_z_previous, model)
    
                if similarity_val >= similarity_th:
                    z["exist"] = True
                    z["vanish_counter"] = 0
                    z["duration"] += 1
                    z["state"] = 1

                else:
                    z["exist"] = False
                    z["vanish_counter"] += 1
                    z["duration"] = 0
                    z["state"] = -1

                if z["vanish_counter"] >= vanish_th:
                    remove_idx.append(z_idx)

            # 更新 Tracking List 各個物件的 ref 等等參數
            # 1. 刪除
            for r_idx in remove_idx[::-1]:
                # remove_idx 中的物件必為 Tracking List 的子集合 
                del tracking_list[r_idx]

            # 2. 新增
            for add_obj in wait_to_add:
                tracking_list.append(add_obj)

            # 3. 更新參數
            for obj_idx, obj in enumerate(tracking_list):
                tracking_list[obj_idx]["ref"] = False

            # 設置上一幀
            previous_rgb_frame = rgb_frame.copy()

            # Draw bbox----------------------------------------- #
            for tracking_obj in tracking_list:

                if tracking_obj["state"] == 1: # 追蹤中
                    color = (0, 0, 255)
                elif tracking_obj["state"] == 0: # 剛進
                    color = (0, 255, 0)
                elif tracking_obj["state"] == -1: # 遺失
                    color = (126, 0, 0)

                tmp_bbox = tracking_obj["bbox"]
                cv2.rectangle(denoise_mask, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)
                cv2.rectangle(rgb_frame, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)

            cv2.imwrite(f"{path2output}/mask/{current_frame_idx}.png", denoise_mask)
            cv2.imwrite(f"{path2output}/rgb/{current_frame_idx}.png", rgb_frame)
            # -------------------------------------------------- #
                
            print(f"Current frame: {current_frame_idx}, TrackingList Size is: {len(tracking_list)}")          

        rgb_cap.release()
        bgs_cap.release()
        

if __name__ == "__main__":
    main_copy()
    # test()