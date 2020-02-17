import os
import numpy as np
import cv2


path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-17-47/5'
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-07-29--11-17-20/4'


def remove_previous_time(input, cut_time):
    print("Before : ", len(input))
    for j, data in enumerate(input):
        time_delta = input[j][0][0] - input[0][0][0]
        # print(time_delta)
        if(time_delta >= cut_time):
            print("After : ", len(input[j:]))
            return input[j:]


if __name__ == "__main__":
    ### Load preview png img
    # preview_img = cv2.imread(os.path.join(path, 'preview.png'))
    # cv2.imshow('preview_img', preview_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    ### Load car_speed
    log_path = os.path.join(path, 'processed_log', 'CAN', 'speed')
    CAN_time = np.load(os.path.join(log_path, 't'))
    value = np.load(os.path.join(log_path, 'value'))
    time_expand = np.expand_dims(CAN_time, axis=1)
    time_and_value = np.stack((time_expand, value), axis=2)
    CAN_len = time_and_value.shape[0]
    CAN_time = time_and_value[-1][0][0] - time_and_value[0][0][0]
    print("Total CAN data length : ", CAN_len)
    print("Total Time of CAN data : ", CAN_time)

    CAN_per_sec = CAN_len / CAN_time
    CAN_delay = 1000.0 / CAN_per_sec
    CAN_idx_unit = CAN_per_sec / 25.0
    print("CAN_per_sec : ", CAN_per_sec)
    print("CAN_delay (ms) : ", CAN_delay)
    print("CAN_idx_unit : ", CAN_idx_unit)

    '''
    78642.50749 ~ 78702.50153 == 59.994sec ~= 60sec
    1200 Frames , (25 FPS) => Total 48 sec
    CAN data / video == 60/48 = 5/4 = 1.25
    '''

    cyan = (0, 0, 255)
    thickness = 2
    location = (30, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5

    ### Load hevc encoded video
    hevc_video = cv2.VideoCapture(os.path.join(path, 'video.hevc'))
    fps = hevc_video.get(cv2.CAP_PROP_FPS)
    delay = round(1000 / fps) # 40ms
    total_video_sec = 48.0
    print("fps :", fps) # 25.0
    print("delay (ms) :", delay)
    total_delay = 0
    i = 0
    j = 0

    # Remove previous time
    # print(time_and_value[0][0][0], time_and_value[-1][0][0])
    # print(time_and_value[-1][0][0] - time_and_value[0][0][0])
    # exit()
    time_and_value = remove_previous_time(time_and_value, cut_time=12)

    while(hevc_video.isOpened()):
        ret, frame = hevc_video.read()
        if ret == True:
            ## Time
            frame_time = total_delay / 1000

            # CAN Time
            CAN_time = time_and_value[j][0][0] - time_and_value[0][0][0]
            CAN_time = round(CAN_time, 4)
            speed = time_and_value[j][0][1]
            speed = round(speed, 4)
            cv2.putText(frame, f'F_t: {frame_time} || CAN_t: {CAN_time} || Speed: {speed}', location, font, fontScale,
                        cyan, thickness)

            ## Show
            cv2.imshow('frame', frame)
            i += 1
            j += int(round(CAN_idx_unit))
            total_delay += delay
            if cv2.waitKey(delay) & 0xFF == ord('q'):
            # if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    hevc_video.release()
    cv2.destroyAllWindows()
    print("Total Frame : ", i)
    print("Total Time (sec) : ", total_delay / 1000)