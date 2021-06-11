import cv2
import numpy as np
from keras.models import load_model

x = int()
y = int()
model = load_model('Gender_classification.h5')

def prediction(image):
    image = cv2.resize(image,(150,150))
    image = image.reshape(1,150,150,3)
    image = image/255
    dic = {0:'Female',1:'Male'}
    result = model.predict(image)
    result = np.argmax(result)
    return dic[result]

# open the camera
cap = cv2.VideoCapture(0)
# load the face detection file
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if (cap.isOpened()== False):
    print("Error opening video stream or file")
color = (0,0,255) # (b,g,r)
# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    # if ret is true then  frame has image in nd.array format
    if ret == True:
        # getting the label
        label = prediction(frame)
        # getting the co-ordintes of face
        bboxes = classifier.detectMultiScale(frame)
        
        for box in bboxes:
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # put the rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x2, y2), (0,0,255))
            # putting the label in captured image (frame) 
        text = cv2.putText(frame, label,(x-1,y-1),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)    
       
        # display the frame
        cv2.imshow("GENDER CLASSIFICATION", text)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        print('something went wrong')    



cap.release()
cv2.destroyAllWindows()