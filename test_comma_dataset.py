import os
import numpy as np
import cv2

# raw_path = '/home/titan/hdd_ext/hdd2/comma2k19_ext'

### Load preview png img
path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
# preview_img = cv2.imread(os.path.join(path, 'preview.png'))
# cv2.imshow('preview_img', preview_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### Load car_speed
path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
log_path = os.path.join(path, 'processed_log', 'CAN', 'speed')
t = np.load(os.path.join(log_path, 't'))
value = np.load(os.path.join(log_path, 'value'))
# print(t.shape) # (4975,)
# print(value.shape) # (4975, 1)
# print(value)
print(t[0], value[0,0], value[0])
t_expand = np.expand_dims(t, axis=1)
# print(t_expand.shape)
t_and_value = np.stack((t_expand, value), axis=2)
print(t_and_value.shape)
print(t_and_value[0])
print(t_and_value[-1])

'''
78642.50749 ~ 78702.50153 == 59.994sec ~= 60sec
1200 Frames , (25 FPS) => Total 48 sec
CAN data / video == 60/48 = 5/4 = 1.25
'''

### Load hevc encoded video
# video_path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
hevc_video = cv2.VideoCapture(os.path.join(path, 'video.hevc'))
fps = hevc_video.get(cv2.CAP_PROP_FPS)
print("fps :", fps) # 25.0
delay = round(1000 / fps) # 40ms
print("delay :", delay)
i = 0
while(hevc_video.isOpened()):
    ret, frame = hevc_video.read()
    if ret == True:
        cv2.imshow('frame', frame)
        i = i + 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        break
hevc_video.release()
cv2.destroyAllWindows()
print(i) # 1200