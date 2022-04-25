# import tensorflow as tf
# import tensorflow_datasets as tfds
import cv2
import numpy as np
# from tensorflow import keras
# from keras import layers
import os
import joblib
from tqdm import tqdm

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def add_pad(video, pad_size):
    y, x = video.shape[1:3]
    pads = np.zeros((pad_size, y, x, 3), dtype=np.ubyte)
    return np.concatenate([video, pads])

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      # frame = crop_center_square(frame)
      # frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
    frames = np.array(frames, dtype=np.ubyte)
    frames = add_pad(frames, pad_size=40 - len(frames))

  return frames

for folder in tqdm(os.listdir('./data/JHMDB_video/ReCompress_Videos')):
    print(folder)
    videos = np.empty((0, 40, 224, 224, 3), dtype=np.ubyte)
    if folder != '.DS_Store':
        for file in os.listdir('./data/JHMDB_video/ReCompress_Videos/' + folder):
            if file.endswith('.avi'):
                videos = np.append(videos,load_video('./data/JHMDB_video/ReCompress_Videos/' + folder + '/' + file))
                joblib.dump(videos, './data/videos_' + folder +'.pkl')