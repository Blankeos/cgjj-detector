# 🕵️‍♀️ CGJJ Detector

![cgjj preview](/docs/cgjj-preview.gif)

### 🤔 About

The **CGJJ** or the **Carlo Glecy Jessa Jonah Detector** is an object detection model trained on YoloV5 to detect the faces of Carlo, Glecy, Jessa, and Jonah.

We annotated a custom 30FPS video dataset on [makesense.ai](https://makesense.ai/) for an object detection task of 4 classes: `carlo`, `glecy`, `jessa`, `jj`. We have 665 images for training and 100 images for testing.

This repository has two purposes:

1. [Use the model for inference](#🚀-get-started)
2. [Recreate the model from scratch](#😎-how-to-recreate-this-app-from-scratch)

Submitted to **Mr. John Christopher Mateo** for **CCS 250 Computer Vision** as a Final Project for the semester.

### 🚀 How to Use:

1. Clone this repository

```sh
$ git clone https://www.github.com/blankeos/cgjj-detector
$ cd cgjj-detector
```

2. Install the requirements

```sh
$ python -m venv venv # Optional: Create a virtual environment
$ pip install -r requirements.txt
```

3. Run scripts to use model for inference (You have 3 choices):

   - [x] Create video from `./videoData/test.mp4` (Smoothest but Long Process Time)

   ```sh
   python create_vid.py
   # Saves video on 'output_video.mp4'
   ```

   - [x] Use model in real-time on **camera** (Choppy depending on GPU)

   ```sh
   python realtime_cam.py
   ```

   - [x] Use model in real-time on `./videoData/test.mp4` (Choppy depending on GPU)

   ```sh
   python realtime_vid.py
   ```

### 📝 Important Links

- [Final Annotated Dataset](https://carlo.vercel.app/)
- [YoloV5 Training Notebook](https://carlo.vercel.app/)
- [Final Model Used](https://carlo.vercel.app/)

### 📁 Dataset Directory Structure

```
| - dataset
    | -- images
        | -- train
        | -- val
    | -- labels
        | -- train
        | -- val
    labels.txt
```

### 🧠 Final Model Performance

```
Model summary: 157 layers, 7020913 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  2.30it/s]
                   all        100        400      0.996      0.997      0.995      0.775
                 carlo        100        100          1       0.99      0.995      0.789
                 glecy        100        100      0.998          1      0.995      0.689
                 jessa        100        100      0.999          1      0.995      0.815
                    jj        100        100      0.988          1      0.995      0.805
```

---

### 😎 How to recreate this project from scratch

Make sure you already cloned this repository and installed the requirements.

1. Run preprocessing script

```sh
python preprocessing.py
# This will use `./videoData` and create images following the YoloV5 directory structure
# Then, outputs the preprocessed data on `/cgjj-dataset`
```

2. Annotate the dataset on [makesense.ai](https://makesense.ai/) for an object detection task of 4 classes: `carlo`, `glecy`, `jessa`, `jj`. _(It took us 6 hours to do this)_ 😅
3. Upload the dataset on your Google Drive (To be trained on Colab).
4. Train the dataset on our **YoloV5 Training Notebook**.
5. Download the model from `runs/expt/best_weights.pt` in Colab
6. Put the `best_weights.pt` inside `/model` dir of this repository. Rename as `cgjj_best.pt`.
7. Go back to Step 3 in [Get Started](#🚀-get-started).
