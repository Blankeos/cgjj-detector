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
  print(f"[Generating Frames From Video: {videodir}]")
  while success:
      cv.imwrite(f"{outdir}/frame{count}.jpg", image) # save frame as JPEG file      
      success,image = cap.read()
      print(f'\r+ Read a new frame {count}: {success}'.ljust(30), end="\r")
      count += 1
  print(f"Done: {videodir}. Frames saved to {outdir}.")

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

  print("[Generating Frames with Predictions]")
  for i, batch in enumerate(batcher(filenames, PREDICT_BATCH_SIZE)):
      percent = ((i+1) / float(NUM_BATCHES))
      bars = f"{'█' * int(50 * percent)}".ljust(50, '-')
      print(f"\rPredicting {len(batch)} Frames in Batch {i+1}/{NUM_BATCHES} |{bars}| {(percent*100):.2f}%        ", end="\r")
      filenames_og = [ORIGINAL_PATH + "/" + filename for filename in batch]
      filenames_rd = [RENDERED_PATH + "/" + filename for filename in batch]
      ims = detect(filenames_og)
      
      for i, im in enumerate(ims):
        cv.imwrite(filenames_rd[i], im)
  percent = 1
  bars = f"{'█' * int(50 * percent)}".ljust(50, '-')
  print(f"\rPredicting {len(batch)} Frames in Batch {NUM_BATCHES}/{NUM_BATCHES} |{bars}| 100%        ")
  print(f"Done: Saved to {RENDERED_PATH}")

generate_predictions(filenames)

# 5. Stitch Together to Create a Video
RENDERED_VIDEO_PATH = "output_video.mp4"
width, height = imagesize.get(RENDERED_PATH+"/frame0.jpg") # get any image
frameSize = (width, height)

out = cv.VideoWriter(RENDERED_VIDEO_PATH, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, frameSize)


filenames = natsorted(os.listdir(RENDERED_PATH))

print("[Stitching Images Into Video]")
for filename in filenames:
  img = cv.imread(f"{RENDERED_PATH}/{filename}")
  out.write(img)
print("Done: Final video saved to {RENDERED_VIDEO_PATH}")
out.release()


# Open Video
from os import startfile
startfile(RENDERED_VIDEO_PATH)