import os
import numpy as np
import cv2
import parmap

data_path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/'
output_dir_path = os.path.join('/home/titan/hdd_ext/hdd2', 'comma2k19_preprocessed')
night_threshold = 40.0


def remove_none_dir(chunk, sub_chunk_list):
    for sub_chunk in sub_chunk_list:
        if not os.path.isdir(os.path.join(data_path, chunk, sub_chunk)):
            print("No Dir:", sub_chunk)
            sub_chunk_list.remove(sub_chunk)
        else:
            pass
    return sub_chunk_list


def extract_frame(input_path, image_path, only_day_scene=True):
    path, file_name = os.path.split(input_path)

    preview_path = os.path.join(path, 'preview.png')
    preview_img = cv2.imread(preview_path)
    if np.mean(preview_img) < night_threshold:
        return

    path = path.split(data_path)
    path = path[1]
    extract_path = os.path.join(image_path, path)
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


if __name__ == "__main__":
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    image_path = os.path.join(output_dir_path, 'input')
    masked_image_path = os.path.join(output_dir_path, 'masked_images')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(masked_image_path):
        os.mkdir(masked_image_path)

    total_video_list = []
    chunk_list = os.listdir(data_path)
    for chunk in chunk_list:
        sub_chunk_list = os.listdir(os.path.join(data_path, chunk))
        sub_chunk_list = remove_none_dir(chunk, sub_chunk_list)
        for sub_chunk in sub_chunk_list:
            video_list = os.listdir(os.path.join(data_path, chunk, sub_chunk))
            # print(video_list)
            for video in video_list:
                video_path = os.path.join(data_path, chunk, sub_chunk, video, 'video.hevc')
                total_video_list.append(video_path)
    total_video_list.sort()

    ## Multi-processing
    num_cores = 10
    parmap.map(extract_frame, total_video_list, image_path, True, pm_pbar=True, pm_processes=num_cores)
