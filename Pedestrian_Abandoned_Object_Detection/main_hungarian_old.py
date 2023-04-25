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
from scipy.optimize import linear_sum_assignment

from natsort import natsorted
import yh as yh


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

def checking(box1, box2, overlap_th):  # 檢查若 1.長寬差兩倍以上 2.重疊程度在overlap_th以下，返回false 即代表不是同個物件
    check = 1
    # 若長or寬兩倍以上，不會認定是同物體
    if box1[2]/box2[2] > 2 or box2[2]/box1[2] > 2 or box1[3]/box2[3] > 2 or box2[3]/box1[3] > 2:
        check = 0
    # 若iou小於閥值（重疊太少），不是同物品
    if yh.iou(box1, box2) < overlap_th:
        check = 0

    return check

def get_iou(box1, box2):
    # tmpp = yh.iou(box1, box2)
    # print(f'tmpp:{tmpp}')
    return yh.iou(box1, box2)

def check_sizediff(box1, box2): 
    if box1[2]/box2[2] > 2 or box2[2]/box1[2] > 2 or box1[3]/box2[3] > 2 or box2[3]/box1[3] > 2:
        return True

    
def get_similarity(box1, box1_rgb, box2, box2_rgb, model):
    height, width, _ = box1_rgb.shape
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ul_x, ul_y, lr_x, lr_y = yh.dilate_bbox(box1, height, width)
    y_cropped_img = box1_rgb[ul_y:lr_y, ul_x:lr_x]

    ul_x, ul_y, lr_x, lr_y = yh.dilate_bbox(box2, height, width)
    x_cropped_img = box2_rgb[ul_y:lr_y, ul_x:lr_x]

    tensor_y_cropped_img, tensor_x_cropped_img = transform(y_cropped_img).unsqueeze(0), transform(x_cropped_img).unsqueeze(0)
    sim_score = similarity(tensor_y_cropped_img, tensor_x_cropped_img, model)
    return sim_score

def hungarian_algorithm(cost_matrix):
    n, m = cost_matrix.shape
    print(cost_matrix.shape)
    matches = np.full(n, -1, dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i in range(len(row_ind)):
        matches[row_ind[i]] = col_ind[i]
    return matches

def main():    

    # 所有流程的中間產物會放到 "./our_datum/tmp" 底下
    path2tmp = "./our_datum1/tmp"
    if not os.path.exists(path2tmp): os.makedirs(path2tmp)

    # 1(a). 把監視器影片放到 ./our_datum/input 底下
    # 注意! 目前一次只能轉一部
    path2input = "./our_datum1/input"
    # video_name = os.listdir(path2input)[0]
    # path2video = f"{path2input}/{video_name}"
    # if not os.path.exists(path2video): 
    #     print(f"Fail to read any video at {path2video}.")
    #     os._exit(0)
    # # 1(b). 儲存影片每幀到 ./our_datum/tmp/rgb_frames
    # yh.video2frames(path2video=path2video, save_path=f"{path2tmp}/rgb_frames")


    # # 2(a). 對監視器影片做 Yolo Seg 將mask儲存成圖片，存放至 ./our_datum/tmp/yolo_seg
    # if not os.path.exists(f"{path2tmp}/yolo_seg"): 
    #     print(f"Fail to read img in {path2tmp}/yolo_seg.")
    #     os._exit(0)
    # # 2(b). 對監視器影片做 vibe 存放至 ./our_datum/tmp/vibe.mp4
    # # Skip...
    # # 2(c). 儲存 vibe 的每1幀 存放至 ./our_datum/tmp/vibe
    # yh.video2frames(path2video=f"{path2tmp}/vibe.mp4", save_path=f"{path2tmp}/vibe")


    # # 3(a). 做 bitwise_or 將結果儲存至 ./our_datum/tmp/bitwise_or
    # yh.bitwise_process_or(f"{path2tmp}/vibe", f"{path2tmp}/yolo_seg", f"{path2tmp}/bitwise_or")
    # # 3(b). 將 bitwise_or 結果轉成影片儲存
    # yh.frames2video(f"{path2tmp}/bitwise_or", path2tmp, save_name="aft_bitwise.mp4", get_fps_from=path2video)


    # # 4. 降躁並取得連通域訊息
    # yh.denoise_on_bin_video(path2video=f"{path2tmp}/aft_bitwise.mp4", save_path=path2tmp, save_name="aft_denoise.mp4", npy_save_path=f"{path2tmp}/npys", area_threshold=800, save_bbox=True)
    

    # 5.Tracking

    # ----------user_config---------- #
    
    # thresholds
    iou_th = 0.5
    similarity_th = 0.9
    duration_th = 10
    vanish_th = 5
    stay_up_th = 250    # 存在超過此幀數，即使在前景消失也不會被消除
    overlap_th = 0.3    # iou沒達到此閥值不會是同物體
    delete_cnt = 5      # 連續幾幀沒被配到前景就會被刪掉

    # other
    path2output = "./our_datum1/output"
    if not os.path.exists(f"{path2output}/mask"): os.makedirs(f"{path2output}/mask") 
    if not os.path.exists(f"{path2output}/rgb"): os.makedirs(f"{path2output}/rgb")

    path2frames, path2npy, path2mask = f"{path2tmp}/rgb_frames", f"{path2tmp}/npys", f"{path2tmp}/bitwise_or"
    frames, npys, masks = natsorted( os.listdir(path2frames) ), natsorted( os.listdir(path2npy) ), natsorted( os.listdir(path2mask) )

    # 線的厚度, 類型
    thickness, lineType = 2, 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ------------------------------ #

    model = load_model()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # initialize tracking list
    tracking_list = []
    # Main Loop
    for current_idx, (frame, npy) in enumerate(zip(frames, npys)):

        # 設定影片範圍
        if current_idx <= 499: 
            continue
        if current_idx >= 700:
            break

        # # # 設定影片範圍
        # if current_idx <= 20: 
        #     continue
        # if current_idx >= 200:
        #     break


        now_rgb_frame, interest_bboxes = cv2.imread( f"{os.path.join(path2frames, frame)}" ), np.load( f"{os.path.join(path2npy, npy)}" )
        pre_rgb_frame = cv2.imread(f"{path2frames}/{frames[current_idx-1]}")

        wait_to_add = [] # wait_to_add 用來儲存加入清單之物體

        print(f"frame: {current_idx}, There are {interest_bboxes.shape[0]} obj in interest_bbox")

        ########## HUNGARIAN ####################################################################################################################

        # create cost_matrix
        n = len(interest_bboxes)
        m = len(tracking_list)

        if m == 0:   # tracking list is empty
            for i, y_box in enumerate(interest_bboxes):   
                tmp = {
                        "bbox": y_box,
                        "frame_idx": current_idx,
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

        else:
            cost_matrix = np.zeros((n, m))

            for i, y_box in enumerate(interest_bboxes):      # y 
                y_area = y_box[2]*y_box[3]
                for j, x_box in enumerate(tracking_list):    # x 
                    # get iou
                    iou_score = yh.iou(y_box, x_box["bbox"])

                    # get difference of box size
                    x_area = x_box["bbox"][2]*x_box["bbox"][3]
                    size_score = abs(y_area - x_area) / max(y_area, x_area)

                    # get similarity score
                    sim_score = get_similarity(y_box, now_rgb_frame, x_box["bbox"], pre_rgb_frame, model)

                    # 計算總權重
                    total_score = iou_score + (1 - size_score) + sim_score

                    cost_matrix[i][j] = total_score
                

            max_cost = np.max(cost_matrix) 
            cost_matrix = max_cost - cost_matrix # 原本是越大越好（因為相似度、iou越大越好），要改成越小越好
            match = hungarian_algorithm(cost_matrix)
        



            for i, y_box in enumerate(interest_bboxes): 
                # print(f'for y_box {i}')
                if match[i] == -1:        # y比x多，沒有配到x -> 新物體
                    tmp = {
                        "bbox": y_box,
                        "frame_idx": current_idx,
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
                else:                       # row_ind有x，要檢查有沒有過三關
                    # print(f'y_box: {y_box}')
                    # print(f'match for y: {match[i]}')
                    # print(f'x_box: {tracking_list[match[i]]["bbox"]}')
                    # print(f'iou: {yh.iou(y_box, tracking_list[match[i]]["bbox"])}')
                    # print(f'sizediff: {check_sizediff(y_box, tracking_list[match[i]]["bbox"])}')
                    # print(f'similarity: {get_similarity(y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], pre_rgb_frame, model)}')
                    if (get_iou(y_box, tracking_list[match[i]]["bbox"]) < overlap_th) or check_sizediff(y_box, tracking_list[match[i]]["bbox"]) \
                            or (get_similarity(y_box, now_rgb_frame, tracking_list[match[i]]["bbox"], pre_rgb_frame, model) < similarity_th):
                        # 沒過--> 新物體
                        tmp = {
                            "bbox": y_box,
                            "frame_idx": current_idx,
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
                    else:   # 有配到且通過判斷 --> 更新
                        # print('pass, update')
                        tracking_list[match[i]]["bbox"] = y_box
                        tracking_list[match[i]]["ref"] = True
                        tracking_list[match[i]]["exist"] = True
                        tracking_list[match[i]]["duration"] += 1
                        tracking_list[match[i]]["vanish_counter"] = 0
                        tracking_list[match[i]]["isMoving"] = False
                        tracking_list[match[i]]["state"] = 1
                        tracking_list[match[i]]["no_pair_counter"] = 0

        remove_idx = []
        for z_idx, z in enumerate(tracking_list): # x
            # 有被參考，跳下一個物體 
            if z["ref"] == True: continue
                
            z["no_pair_counter"] += 1
            if z["no_pair_counter"] >= delete_cnt and not z["is_garbage"]:  # 連續delete_cnt幀沒配到前景，且非垃圾，刪除
                remove_idx.append(z_idx)
                continue
            
            # 這邊的x可能是 1.本來tracking_list比較長所以沒有配到y,  2.有配到但是沒有過門檻 --> 跟前一幀的同位子比相似度
            sim_score = get_similarity(z["bbox"], now_rgb_frame, z["bbox"], pre_rgb_frame, model)

            if sim_score >= similarity_th:
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

#########################################################################################################################################
        
        # break
        
    
        
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

        # Draw bbox----------------------------------------- #
        current_bbox_frame = cv2.imread(f"{path2mask}/{masks[current_idx]}")
        for tracking_obj in tracking_list:

            if tracking_obj["state"] == 1: # 追蹤中
                color = (0, 0, 255)
            elif tracking_obj["state"] == 0: # 剛進
                color = (0, 255, 0)
            elif tracking_obj["state"] == -1: # 遺失
                color = (126, 0, 0)

            tmp_bbox = tracking_obj["bbox"]
            cv2.rectangle(current_bbox_frame, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)
            cv2.rectangle(now_rgb_frame, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[0]+tmp_bbox[2]-1, tmp_bbox[1]+tmp_bbox[3]-1), color, thickness, lineType)

        cv2.imwrite(f"{path2output}/mask/{current_idx}.png", current_bbox_frame)
        cv2.imwrite(f"{path2output}/rgb/{current_idx}.png", now_rgb_frame)
        # -------------------------------------------------- #
        
        print(f"Current frame: {current_idx}, TrackingList Size is: {len(tracking_list)}")

if __name__ == "__main__":
    main()