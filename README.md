# CGJJ Detector

<Screenshots>

### 🤔 About

The **CGJJ** or the **Carlo Glecy Jessa Jonah Detector** is an object detection model trained on YoloV5 to detect the faces of Carlo, Glecy, Jessa, and Jonah.

We annotated a custom 30FPS video dataset (600 images) on [makesense.ai](https://makesense.ai/) for an object detection task of 4 classes: `carlo`, `glecy`, `jessa`, `jj`.

Submitted to Mr. John Christopher Mateo for CCS 250 Computer Vision as a Final Project for the semester.

### 🚀 Get Started

1. Clone this repository

```sh
$ git clone https://www.github.com/blankeos/cgjj-detector
$ cd cgjj-detector
```

2. Install the requirements

```sh
$ pip install -r requirements.txt
```

3. Run the app

```sh
python app.py
```

### 😎 How to recreate this app from scratch

1. Refer to the [Important Links](#📝-important-links) section of this docs.
2. Download our **Dataset** video and put inside `/videoData` directory in this repository.
3. Run preprocessing script

```sh
python preprocessing.py
# outputs the preprocessed data on /data
```

4. Annotate the dataset on [makesense.ai](https://makesense.ai/) for an object detection task of 4 classes: `carlo`, `glecy`, `jessa`, `jj`.
5. Upload the dataset on your Google Drive (To be trained on Colab).
6. Train the dataset on our **YoloV5 Training Notebook**.
7. Export the model from `runs/expt3/best_weights.pt`
8. Put the `best_weights.pt` inside `/model` dir of this repository.
9. Run the app

```
python app.py
```

### 📝 Important Links

- [Dataset](https://carlo.vercel.app/)
- [YoloV5 Training Notebook](https://carlo.vercel.app/)
- [Final Model Used](https://carlo.vercel.app/)
