#Import the required libraries.
import os
import cv2
#import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
#from moviepy.editor import
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from moviepy.editor import VideoFileClip


'''For this stage, you may have to upload the test zip file added in the folder'''

dataset_path = os.listdir('/train')

CLASSES_LIST = os.listdir('/test')

#get model for use
LRCN_model = tf.model.load("model", None)

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    """
    This function will perform single action recognition prediction on a video using the LRCN model.

    Args:
    video_file_path: The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH: The fixed number of frames of a video that can be passed to the model as one sequence.
    """

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the frame to fixed dimensions.
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()


'''Here I normalize the pixels, set the image height as well
  as the legnth of each video sequence'''
max_pixel_value = 255
image_height, image_width=64,64
SEQUENCE_LENGTH=100

'''Here, i extract the frames, iterate through them, resize based on the the set
  image size and normalize i.e 0-1 rather than 0-255. Everything is added to a list'''

def frames_extraction(video_path):
  frames_list = []

  #print(" the video file path is : {}".format(video_path))
  videoObj = cv2.VideoCapture(video_path)
  #print("the video object is: {}".format(videoObj))

  """ Iterating through Video Frames """
  while True:

    # Reading a frame from the video file
    success, image = videoObj.read()
    #print("the value of success is: {}".format(success))

    if not success:
      break

    resized_frame = cv2.resize(image, (image_height, image_width))

    """Normalize the resized frame by dividing it with 255 so that
    each pixel value then lies between 0 and 1"""

    normalized_frame = resized_frame / max_pixel_value
    frames_list.append(normalized_frame)


  videoObj.release()
  return frames_list


# Makes Prediction
input_video_file_path = "C:\\Users\\ben4s\\Desktop\\my computer vision\\RNN-CNN video classifier\\test\\walking\\girl-walking-on-university-campus_b1s8vlgqr__8123088590bd8669ae28e877270ca090__P360.mp4"

# Perform single prediction on the test video.
predict_single_action(input_video_file_path, SEQUENCE_LENGTH)

# Display the input video.
VideoFileClip(input_video_file_path, audio=False, target_resolution=(300, None)).ipython_display()