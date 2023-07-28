# import the opencv library
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}


# define a video capture object
cap1 = cv2.VideoCapture(0)
show_text=[0]
size = 100


while(True):
	
	# Capture the video frame
	# by frame
	flag1, frame1 = cap1.read()
	frame1 = cv2.resize(frame1, (600, 500))
	
	bounding_box = cv2.CascadeClassifier("D:\\Program Files\\Users\\Jayesh\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
	gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
	for(x, y, w, h) in num_faces:
		cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
		roi_gray_frame = gray_frame[y:y + h, x:x + w]
		cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
		prediction = emotion_model.predict(cropped_img)
		print(prediction)
		maxindex = int(np.argmax(prediction))
		print(maxindex)
		cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		show_text[0]=maxindex
		emoji = cv2.imread(emoji_dist[show_text[0]])
		emoji = cv2.resize(emoji, (size, size))
		img2gray = cv2.cvtColor(emoji, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
		roi = frame1[-size-10:-10, -size-10:-10]
		roi[np.where(mask)] = 0
		roi += emoji
	# Display the resulting frame
	cv2.imshow('frame', frame1)
	
    
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
cap1.release()
# Destroy all the windows
cv2.destroyAllWindows()
