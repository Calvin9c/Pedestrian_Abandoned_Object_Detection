import argparse

def tracking_config():
    parser = argparse.ArgumentParser(description='Configure the thresholds of tracking phase.')
    parser.add_argument('--similarity_th', type=float, default=0.8, help='相似度達閥值當作同一物品')
    parser.add_argument('--high_iou_th', type=float, default=0.98, help='iou達此閥值的話即使相似度低也沒關係')
    parser.add_argument('--high_iou_simi_th', type=float, default=0.7, help='iou高時相似度閥值可較低')
    parser.add_argument('--stay_up_th', type=int, default=400, help='存在超過此幀數，有可能是垃圾')
    parser.add_argument('--overlap_th', type=float, default=0.3, help='iou沒達到此閥值不會是同物體')
    parser.add_argument('--delete_cnt', type=int, default=5, help='連續幾幀沒被配到前景就會被刪掉')
    parser.add_argument('--covered_th', type=float, default=0.8, help='被覆蓋的面積佔自己的比例超過此閥值認定為被覆蓋')
    parser.add_argument('--moved_th', type=int, default=200, help='累積移動量超過此閥值不會是垃圾')
    parser.add_argument('--be_covered_time_th', type=int, default=300, help='累積被覆蓋了幀數超過此閥值，會被更新/刪掉')
    parser.add_argument('--sizediff_yolo_th', type=int, default=2, help='和yolo框的大小差距小於此閥值，是鬼影')
    parser.add_argument('--reset_cnt', type=int, default=30, help='le of tracking list達到此數字，清空')

    return parser.parse_args()

def get_cfg(video_name):
    cfg = {
        'frame_interval'        : 2, # 介於1~3之間，2為最佳。
        'run_yolo_only'         : True,
        'debug_mode'            : False,

        'video_name'            : video_name,
        'save_path_for_dict'    : f"{video_name}/dict",
        'save_pth_for_res_cc'   : f"{video_name}/cc_res_npys",
        'save_pth_for_del_cc'   : f"{video_name}/cc_del_npys",
        'save_pth_for_yolo_box' : f"{video_name}/yolo_npys",

        'sec_proc_area_th'      : 200, # lightweight_denoise
        'trd_proc_area_th'      : 800, # denoise

        'denoise_fst_proc'      : f"{video_name}/debug/denoise/first_process",
        'denoise_sec_proc'      : f"{video_name}/debug/denoise/second_process",
        'denoise_trd_proc'      : f"{video_name}/debug/denoise/third_process"
    }
    assert 1 <= cfg['frame_interval'] and cfg['frame_interval'] <= 3
    return cfg