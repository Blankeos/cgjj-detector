import torch
import cv2 as cv

VIDEO_PATH = "./videoData/test.mp4"

# Model
model = torch.hub.load('./yolov5', 'custom', path='./model/cgjj_best.pt', source='local')  # local repo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Draw on Image
def detect(image):
    results = model(image)
    results.render()
    return results.ims[0]

# -- VIDEO CAPTURE --
cap = cv.VideoCapture(VIDEO_PATH)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv.imshow('Frame',detect(frame))
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break