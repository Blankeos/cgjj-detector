import torch
import cv2 as cv

# Model
model = torch.hub.load('./yolov5', 'custom', path='./model/cgjj_best.pt', source='local')  # local repo

# Draw on Image
def detect(image):
    results = model(image)
    results.render()
    return results.ims[0]

# define a video capture object
vid = cv.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    frame = detect(frame)
    # Display the resulting frame
    cv.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()