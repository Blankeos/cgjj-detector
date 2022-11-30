import torch
import cv2 as cv


# Model
model = torch.hub.load('./yolov5', 'custom', path='./model/cgjj_best.pt', source='local')  # local repo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Draw on Image
def detect(image):
    results = model(image)
    results.render()
    ims = [cv.cvtColor(im, cv.COLOR_BGR2RGB) for im in results.ims] # convert to RGB
    return ims

import os
import imagesize
from natsort import natsorted

VIDEO_PATH = "./videoData/test.mp4"
PREDICT_BATCH_SIZE = 50

# 1. Create root "temp"
TEMP_ROOT = "./temp"
ORIGINAL_PATH = TEMP_ROOT + "/original"
RENDERED_PATH = TEMP_ROOT + "/rendered"

if not os.path.exists(TEMP_ROOT):
  os.makedirs(TEMP_ROOT)
if not os.path.exists(ORIGINAL_PATH):
  os.makedirs(ORIGINAL_PATH)
if not os.path.exists(RENDERED_PATH):
  os.makedirs(RENDERED_PATH)

# 2. Read Video and Save to TEMP_ORIGINAL
def generate_images_from_video(videodir, outdir):
  cap = cv.VideoCapture(videodir)
  success,image = cap.read()
  count = 0
  while success:
      cv.imwrite(f"{outdir}/frame{count}.jpg", image) # save frame as JPEG file      
      success,image = cap.read()
      print(f'Read a new frame {count}: ', success)
      count += 1

def rrmdir(path):
    for entry in os.scandir(path):
        if entry.is_dir():
            rrmdir(entry)
        else:
            os.remove(entry)
    os.rmdir(path)

generate_images_from_video(VIDEO_PATH, ORIGINAL_PATH)

# 3. Read TEMP_ORIGINAL and Generate Predictions
filenames = natsorted(os.listdir(ORIGINAL_PATH))
filenames_original = [ORIGINAL_PATH + "/" + filename for filename in filenames]
filenames_rendered = [RENDERED_PATH + "/" + filename for filename in filenames]

# 4. Generate Predictions
from itertools import islice

def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def generate_predictions(filenames):
  NUM_BATCHES = len(list(batcher(filenames, PREDICT_BATCH_SIZE)))

  for i, batch in enumerate(batcher(filenames, PREDICT_BATCH_SIZE)):
      print(f"Predicting {i+1}/{NUM_BATCHES} Batch...")
      filenames_og = [ORIGINAL_PATH + "/" + filename for filename in batch]
      filenames_rd = [RENDERED_PATH + "/" + filename for filename in batch]
      ims = detect(filenames_og)
      
      for i, im in enumerate(ims):
        cv.imwrite(filenames_rd[i], im)

generate_predictions(filenames)

# 5. Stitch Together to Create a Video
RENDERED_VIDEO_PATH = "output_video.mp4"
width, height = imagesize.get(RENDERED_PATH+"/frame0.jpg") # get any image
frameSize = (width, height)

out = cv.VideoWriter(RENDERED_VIDEO_PATH, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, frameSize)


filenames = natsorted(os.listdir(RENDERED_PATH))

print("Stitching Images Into Video...")
for filename in filenames:
  img = cv.imread(f"{RENDERED_PATH}/{filename}")
  out.write(img)

out.release()


# Open Video
from os import startfile
startfile(RENDERED_VIDEO_PATH)