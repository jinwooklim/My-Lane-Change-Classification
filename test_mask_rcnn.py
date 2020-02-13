from keras.models import load_model
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
import cv2
import numpy as np
import os

dir_name = os.path.join(os.getcwd(),'comma2k19_test')
image_path = os.path.join(dir_name, 'input')
masked_image_path = os.path.join(dir_name, 'masked_images')
if not os.path.exists(image_path):
    os.mkdir(image_path)
if not os.path.exists(masked_image_path):
    os.mkdir(masked_image_path)

path = '/home/titan/hdd_ext/hdd2/comma2k19_ext/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/1'
hevc_video = cv2.VideoCapture(os.path.join(path, 'video.hevc'))
id = 0
print("Read Video ...")
while(hevc_video.isOpened()):
    ret, frame = hevc_video.read()
    if ret == True:
        # print(np.shape(frame))
        # frame[350:450, 482:682, :] = 0
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(image_path, '%06d.png'%(id)), frame)
        id = id + 1
    else:
        break
hevc_video.release()
print("End of Read")

masked_image_extraction = DetectObjects(image_path, masked_image_path)
masked_image_extraction.save_masked_images()

print("End")

# model = load_model('mask_rcnn_coco.h5')
# model.predict





