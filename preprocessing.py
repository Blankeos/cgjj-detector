import os
import cv2 as cv

# 1. Set Parameters
OUTPUT_DIR = "dataset"
TRAIN_DIR = "./videoData/train.mp4"
VAL_DIR = "./videoData/test.mp4"
LABELS = ["carlo", "glecy", "jessa", "jj"]

# 2. Create `/dataset` directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 3. Create Train and Test Features (Images)
def generate_images_from_images(videodir, outdir, limit=None):
    print(f"Creating {outdir}...")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    vidcap = cv.VideoCapture(videodir)
    success,image = vidcap.read()
    count = 0
    while success:
        cv.imwrite(f"{outdir}/frame{count}.jpg", image) # save frame as JPEG file      
        success,image = vidcap.read()
        print(f'Read a new frame {count}: ', success)
        count += 1
        if (limit != None):
            if (count >= limit):
                break

generate_images_from_images(TRAIN_DIR, f"{OUTPUT_DIR}/images/train")
generate_images_from_images(VAL_DIR, f"{OUTPUT_DIR}/images/val", limit=100)

# 4. Labels.txt
with open(f"{OUTPUT_DIR}/labels.txt", 'w') as f:
    for i, label in enumerate(LABELS):
        f.write(f"{label}")
        if (i != len(LABELS)-1):
            f.write("\n")
    f.close()