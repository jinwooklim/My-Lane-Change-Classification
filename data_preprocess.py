import os
import numpy as np
import cv2
import parmap

dataset_path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/'
output_dir_path = os.path.join('/home/titan/hdd_ext/hdd2', 'comma2k19_preprocessed')
night_threshold = 40.0
video_fps = 25.0
video_frames = 1200
video_running_time = 48.0 # second


def remove_none_dir(chunk, sub_chunk_list):
    for sub_chunk in sub_chunk_list:
        if not os.path.isdir(os.path.join(dataset_path, chunk, sub_chunk)):
            print("No Dir:", sub_chunk)
            sub_chunk_list.remove(sub_chunk)
        else:
            pass
    return sub_chunk_list


def extract_frame(input_path, extract_path, night_threshold):
    path, file_name = os.path.split(input_path)

    preview_path = os.path.join(path, 'preview.png')
    preview_img = cv2.imread(preview_path)
    if np.mean(preview_img) < night_threshold:
        return
    else:
        path = path.split(dataset_path)
        path = path[1]
        extract_path = os.path.join(extract_path, path)
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        p = path.split('/')
        print("Read Video : %s || %s || %s"%(p[0], p[1], p[2]))
        hevc_video = cv2.VideoCapture(input_path)
        id = 0
        while(hevc_video.isOpened()):
            ret, frame = hevc_video.read()
            if ret == True:
                cv2.imwrite(os.path.join(extract_path, '%06d.png'%(id)), frame)
                id += 1
            else:
                break
        hevc_video.release()
        return


def extract_CAN_speed(input_path, extract_path):
    def _remove_previous_time(input, cut_time):
        for j, data in enumerate(input):
            time_delta = input[j][0][0] - input[0][0][0]
            if (time_delta >= cut_time):
                return input[j:]

    def _remove_non_synced_data(input, CAN_idx_unit):
        idx = 0.0
        mask = np.zeros(len(input), dtype=bool)
        idx_list = []
        for i in range(len(input)):
            if int(idx) >= len(input):
                break
            else:
                idx_list.append(int(idx))
                idx += CAN_idx_unit
        mask[idx_list] = True
        synced = input[mask]
        synced = synced[:video_frames]
        return synced

    path, file_name = os.path.split(input_path)
    path = path.split(dataset_path)
    path = path[1]
    extract_path = os.path.join(extract_path, path)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    p = path.split('/')
    print("Read CAN : %s || %s || %s"%(p[0], p[1], p[2]))

    CAN_time = np.load(os.path.join(input_path, 't'))
    value = np.load(os.path.join(input_path, 'value'))
    time_expand = np.expand_dims(CAN_time, axis=1)
    time_and_speed = np.stack((time_expand, value), axis=2)

    CAN_len = time_and_speed.shape[0]
    CAN_time = time_and_speed[-1][0][0] - time_and_speed[0][0][0]

    CAN_per_sec = CAN_len / CAN_time
    CAN_idx_unit = CAN_per_sec / video_fps

    time_and_speed = _remove_previous_time(time_and_speed, cut_time=6)
    time_and_speed = _remove_non_synced_data(time_and_speed, CAN_idx_unit)
    np.save(os.path.join(extract_path, 'time_and_speed'), time_and_speed)
    return


if __name__ == "__main__":
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    extract_frames_path = os.path.join(output_dir_path, 'input_images')
    if not os.path.exists(extract_frames_path):
        os.mkdir(extract_frames_path)
    extract_CAN_path = os.path.join(output_dir_path, 'input_CAN')
    if not os.path.exists(extract_CAN_path):
        os.mkdir(extract_CAN_path)
    # masked_image_path = os.path.join(output_dir_path, 'masked_images')
    # if not os.path.exists(masked_image_path):
    #     os.mkdir(masked_image_path)

    total_video_path_list = []
    total_CAN_speed_path_list = []
    chunk_list = os.listdir(dataset_path)
    for chunk in chunk_list:
        sub_chunk_list = os.listdir(os.path.join(dataset_path, chunk))
        sub_chunk_list = remove_none_dir(chunk, sub_chunk_list)
        for sub_chunk in sub_chunk_list:
            sequence_list = os.listdir(os.path.join(dataset_path, chunk, sub_chunk))
            for sequence in sequence_list:
                video_path = os.path.join(dataset_path, chunk, sub_chunk, sequence, 'video.hevc')
                total_video_path_list.append(video_path)
                CAN_speed_path = os.path.join(dataset_path, chunk, sub_chunk, sequence, 'processed_log', 'CAN', 'speed')
                total_CAN_speed_path_list.append(CAN_speed_path)
    total_video_path_list.sort()
    total_CAN_speed_path_list.sort()

    ## Multi-processing
    num_cores = 10
    parmap.map(extract_frame, total_video_path_list, extract_frames_path, night_threshold, pm_pbar=True, pm_processes=num_cores)
    parmap.map(extract_CAN_speed, total_CAN_speed_path_list, extract_CAN_path, pm_pbar=True, pm_processes=num_cores)