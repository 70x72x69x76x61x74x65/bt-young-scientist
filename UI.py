import pygame
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import random


screenShot = cv2.imread("sample_Screenshot.png")

trc = []
tlc = []
blc = []
brc = []

def flipv(imgg):

    return  cv2.flip( imgg, 1)

brc_data = open("brc.txt", "r")
blc_data = open("blc.txt", "r")
trc_data = open("trc.txt", "r")
tlc_data = open("tlc.txt", "r")

file = open("BadData.txt", "w")

fileCount = 0

for i in (brc_data, blc_data, trc_data, tlc_data):
    for j in i:
        theArrays = [brc, blc, trc, tlc]
        theArrays[fileCount].append(list(map(int, (j.split(" ")))))

    fileCount += 1

data = []

count = 0
colors = ["b", "c", "k", "y"]

# plt.plot(0, 0, colors[2] + "o")
# plt.plot(1920, 1080, colors[2] + "o")

for i in (blc, trc, tlc, brc):
    for j in i:

        if j[1] > 55:
            pass
        else:
      
        		
        	data.append(j)

    count += 1

data = np.array(data)

clf = KMeans(n_clusters=4)
clf.fit(data)

for i in clf.cluster_centers_:
    print(i)




tlcVal = clf.predict([tlc[0]])
blcVal = clf.predict([blc[0]])
trcVal = clf.predict([trc[0]])


def getDirection(MAX_FRAME, dir):
	# this data is owned by Dmitry Kurtaev
	# His page is https://github.com/dkurt
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	## i have no idea who owns this data
	nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')


	dataDirections = {"TopRight": "TRC", "TopLeft": "TLC", "BottomLeft": "BLC", "BottomRight": "BRC"}

	cap = cv2.VideoCapture("PerfectData{}.avi".format(dataDirections[dir]))

	threshold = 60

	guessedBLC = 0
	guessedTLC = 0
	guessedTRC = 0
	guessedBRC = 0

	Frames = 0
	anotherCount = 0

	for i in range(100):
	    try:
	        ret, img = cap.read()

	        # cv2.imshow("Improved Efficency", img)

	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	        for (x, y, w, h) in faces:

	            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
	            roi_gray = gray[y:y + h, x:x + w]
	            eyes = eye_cascade.detectMultiScale(roi_gray)

	            for (eyeX, eyeY, eyeW, eyeH) in eyes:

	                noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
	                for (Nx, Ny, Nw, Nh) in noses:
	                	cv2.rectangle(img, (Nx, Ny), (Nx + Nw, Ny + Nh), (255, 0, 255), 2)

	                ## if our eye is bellow our nose
	                if eyeY > Ny:
	                	pass
	                cv2.rectangle(img, (x + eyeX, y + eyeY), (x + eyeX + eyeW, y + eyeY + eyeH), (0, 255, 0), 2)

	                eye_roi = img[y + eyeY: y + eyeY + eyeH, x + eyeX:x + eyeX + eyeW]
	                eye_roi = cv2.resize(eye_roi, (100, 100))
	                eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
	                _, eye_roi = cv2.threshold(eye_roi, 40, 255, cv2.THRESH_BINARY_INV)
	              

					
	                contours, hierarchy = cv2.findContours(eye_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

	                for i in contours:

	                    counterX, counterY, w, h = cv2.boundingRect(i)

	                    # print(counterX, counterY)

	                    if counterY > 30 and counterX < 60:

	                        cv2.rectangle(eye_roi, (counterX, counterY), (counterX + w, counterY + h), (255, 0, 0), 2)

	                        file.write(str(counterX) + " " + str(counterY) + "\n")
	                        if anotherCount == MAX_FRAME:
	                            raise ValueError
	                        anotherCount += 1

	                        predict = clf.predict([[counterX, counterY]])

	                        if predict == tlcVal:
	                            guessedTLC += 1
	                            Frames += 1

	                        elif predict == blcVal:
	                            guessedBLC += 1
	                            Frames += 1

	                        elif predict == trcVal:
	                            guessedTRC += 1
	                            Frames += 1

	                        else:
	                            guessedBRC += 1
	                            Frames += 1

	        if Frames == 300:
	            raise ValueError

	        # cv2.imshow("ThE EYe YeS", eye_roi)
	        image = eye_roi

	        k = cv2.waitKey(30) & 0xff

	        if k == 97:
	            raise ValueError


	    except cv2.error as errir:

	    	cap.release()
	    	cv2.destroyAllWindows()

	    	return [[guessedTLC, guessedTRC, guessedBLC, guessedBRC].index(max([guessedTLC, guessedTRC, guessedBLC, guessedBRC])), img]
	    	break


	    except ValueError as e:

	    	cap.release()
	    	cv2.destroyAllWindows()
	    	print (guessedTLC, guessedTRC, guessedBLC, guessedBRC)
	    	return [[guessedTLC, guessedTRC, guessedBLC, guessedBRC].index(max([guessedTLC, guessedTRC, guessedBLC, guessedBRC])), img]
	    	break

	    except NameError as n:
	    	pass


	cap.release()
	cv2.destroyAllWindows()

directons = ["TopRight", "TopLeft", "BottomLeft"]

def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")


pygame.init()


## linux only
x, y = map(int, os.popen("xdpyinfo | awk '/dimensions/{print $2}'").read().split("x"))

clock = pygame.time.Clock()
display = pygame.display.set_mode((x, y))


loop = True

while loop:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			loop = False

	display.fill((255, 255, 255))

	## display code

	display.blit(cvimage_to_pygame(screenShot), [0, 0])

	randInt = random.randrange(0, 3)
	change = getDirection(10, directons[randInt])
	print("loop broken")
	print (x, y)

	try:
		display.blit(cvimage_to_pygame(flipv(change[1])), [x / 2 - 350, y / 2 - 200])
	except BaseException as BE:
		pass

	if change[0] == 0:

		screenShot = screenShot[0:int(y/2), 0:int(x/2)]
		screenShot = cv2.resize(screenShot, (x, y))

	elif change[0] == 1:

		screenShot = screenShot[0:int(y/2), int(x/2):x]
		screenShot = cv2.resize(screenShot, (x, y))

	elif change[0] == 2:

		screenShot = screenShot[int(y/2):y, 0:int(x/2)]
		screenShot = cv2.resize(screenShot, (x, y))

	elif change[0] == 3:

		screenShot = screenShot[0:int(y/2), 0:int(x/2)]
		screenShot = cv2.resize(screenShot, (x, y))
		

	else:
		print(change[0])

	pygame.display.update()
	clock.tick(70)
