#!/usr/bin/env python3.6

import rospy
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tfl
import zipfile
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
#from PIL import Image
from sensor_msgs.msg import Image
import tf2_ros
from utils import label_map_util
import tf
from utils import visualization_utils as vis_util
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import geometry_msgs.msg
#import tf_conversions
import cv2
#cap = cv2.VideoCapture(0)   # if you have multiple webcams change the value to the correct one
global objects
bin_image = None


# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'new_graph'  # change to whatever folder has the new graph
# MODEL_FILE = MODEL_NAME + '.tar.gz'   # these lines not needed as we are using our own model
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 4 # we are only using one class in this example (cloth_mask)

# we don't need to download model since we have our own
# ## Download Model

# In[5]:
#
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tfl.Graph()
with detection_graph.as_default():
    od_graph_def = tfl.GraphDef()
    with tfl.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tfl.import_graph_def(od_graph_def, name='')
    sess = tfl.Session(graph=detection_graph)


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (8, 8)




# In[10]:
def callback(data):
    try:
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(data, "rgb8")
        global frame2
        frame2 = bridge.imgmsg_to_cv2(data, "bgr8")
        #image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        image_np = np.asarray(frame)
        global objects
        with detection_graph.as_default():
            
                
                

            print("new")
            #print(frame)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            global coords
            coords = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            #print(coords)
            #print("boxes:",boxes)
            #print("scores:",scores)
            cv2.imshow('object detection', frame)
            cv2.waitKey(1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
               
                
                
                
                
                
    except CvBridgeError as e:
            print(e)


  
def main(args):
    
    rospy.init_node('custom_webcam', anonymous=True)
    
    # subscribing to /ebot/camera1/image_raw topic which is the image frame of camera
    image_sub = rospy.Subscriber("/prius/front_camera/image_raw", Image, callback, queue_size=1, buff_size=2**32)
    

    rate = rospy.Rate(10)

    try:
        rospy.spin()
        rate.sleep()
        
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


    
if __name__ == '__main__':
    
    main(sys.argv)



