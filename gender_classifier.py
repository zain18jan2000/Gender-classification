import cv2
import numpy as np
from keras.models import load_model

y = int()
x = int()
x2 =int()
y2 = int() 
model = load_model('Gender_class(32x32).h5')

def prediction(image):
    try:
       image = cv2.resize(image,(32,32))
       image = image.reshape(1,32,32,3)
       image = image/255
       dic = {0:'Female',1:'Male'}
       prob = model.predict_proba(image)
       res = model.predict(image)
       prob = "{:.2f}".format(np.amax(res) * 100)
       result = np.argmax(res)
       results = dic[result] + ' ' + str(prob) + '%'
       return results
    except:
        return None   
color = (0,0,255) # (b,g,r)

cap = cv2.VideoCapture(0)
    
# load the face detection file
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if (cap.isOpened()== False):
    print("Error opening video stream or file")
    # Read until video is completed

while(cap.isOpened()):
    ret, frame = cap.read()
    
    # if ret is true then  frame has image in nd.array format
    if ret == True:
        # getting the co-ordintes of face
        bboxes = classifier.detectMultiScale(frame)
        for box in bboxes:
            
            x, y, width, height = box
            x2, y2 = x + width, y + height
            break
        crop_img = frame[y:y2,x:x2]
            
        # getting the label
        label = prediction(crop_img) 
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
    print('Oops! Something went wrong')    

cap.release()
cv2.destroyAllWindows()