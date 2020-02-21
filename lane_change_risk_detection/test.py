import os
import numpy as np
import cv2
import queue
from keras.models import load_model

fixed_frame_num=50
img_height = 480
img_width = 692
img_scale_x = 0.1
img_scale_y = 0.1
delay = 40 # ms
masked_image_path = '/home/titan/hdd_ext/hdd2/comma2k19_sample/masked_images/01'


def _img_read(path):
    img = cv2.imread(path).astype(np.float32)
    img /= 255.0
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.resize(img, (0, 0), fx=img_scale_x, fy=img_scale_y)
    return img


def read_frames(masked_image_path, frame_list, idx):
    cv_list = []
    frames = []
    for i in range(fixed_frame_num):
        # print(idx, idx + i)
        frame = _img_read(os.path.join(masked_image_path, frame_list[idx+i]))
        frames.append(frame)
        cv_list.append(os.path.join(masked_image_path, frame_list[idx+i]))
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0) # for batch 1
    return frames, cv_list


if __name__ == "__main__":
    model = load_model('/hdd_ext/ssd2/PROJECT/My-Lane-Change-Classification/maskRCNN_CNN_lstm_GPU.h5')

    que = queue.Queue()

    frame_list = os.listdir(masked_image_path)
    frame_list.sort()
    num_of_frames = len(frame_list)

    for idx in range(num_of_frames - fixed_frame_num + 1):
        frames, cv_list = read_frames(masked_image_path, frame_list, idx)
        que.put_nowait([frames, cv_list])

        while que.qsize():
            frames, cv_list = que.get_nowait()
            result = model.predict_proba(frames)
            print(' safe | dangerous \n', result)

            img = cv2.imread(cv_list[-1])
            cv2.imshow("Test", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
    cv2.destroyAllWindows()