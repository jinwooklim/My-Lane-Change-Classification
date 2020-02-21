from keras.models import load_model
from lane_change_risk_detection.dataset import DataSet
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
import os
import time

# dir_name = os.path.dirname(__file__)
# dir_name = os.path.dirname(dir_name)
#
# image_path = os.path.join(dir_name, 'data/input/')
# masked_image_path =os.path.join(dir_name, 'data/masked_images/')
#
# masked_image_extraction = DetectObjects(image_path, masked_image_path)
# masked_image_extraction.save_masked_images()

masked_image_path = '/home/titan/hdd_ext/hdd2/comma2k19_sample/masked_images'

data = DataSet()
data.read_video(masked_image_path, option='fixed frame amount', number_of_frames=50, scaling='scale', scale_x=0.1, scale_y=0.1)

start_with_load = time.time()
model = load_model('/hdd_ext/ssd2/PROJECT/My-Lane-Change-Classification/maskRCNN_CNN_lstm_GPU.h5')
start = time.time()
print(' safe | dangerous \n', model.predict_proba(data.video))
print("time1 :", time.time() - start_with_load) # 3.7 sec
print("time2 :", time.time() - start) # 0.8 sec

