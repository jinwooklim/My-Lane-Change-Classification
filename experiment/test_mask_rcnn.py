import os
import sys
import cv2
import numpy as np

from keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects

dir_name = os.path.join('/home/titan/hdd_ext/hdd2', 'comma2k19_sample')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
image_path = os.path.join(dir_name, 'input')
masked_image_path = os.path.join(dir_name, 'masked_images')
if not os.path.exists(image_path):
    os.mkdir(image_path)
if not os.path.exists(masked_image_path):
    os.mkdir(masked_image_path)

## Day Scene
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-17-47/5'

## Night Scene
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/6' # 35.86652295800011
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-50-48/8' # 17
# path ='/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_2/b0c9d2329ad1606b|2018-09-23--23-24-40/10'

## Foggy Scene
# path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-07-29--11-17-20/4' # 80

## Preview Test
# preview_img = cv2.imread(os.path.join(path, 'preview.png'))
# print(np.mean(preview_img)) # 35.86652295800011
# cv2.imshow('preview.png', preview_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()

# hevc_video = cv2.VideoCapture(os.path.join(path, 'video.hevc'))
# id = 0
# print("Read Video ...")
# while(hevc_video.isOpened()):
#     ret, frame = hevc_video.read()
#     if ret == True:
#         # print(np.shape(frame))
#         # frame[350:450, 482:682, :] = 0
#         # cv2.imshow("frame", frame)
#         # cv2.waitKey(0)
#         cv2.imwrite(os.path.join(image_path, '%06d.png'%(id)), frame)
#         id = id + 1
#     else:
#         break
# hevc_video.release()
# print("End of Read")

image_path = '/home/titan/hdd_ext/hdd2/comma2k19_preprocessed/input_images/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
masked_image_path = '/home/titan/hdd_ext/hdd2/comma2k19_sample/masked_images/01'
masked_image_extraction = DetectObjects(image_path, masked_image_path)
masked_image_extraction.save_masked_images()

# model = load_model('mask_rcnn_coco.h5')
# model.predict





