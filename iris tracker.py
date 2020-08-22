# this is an old version

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

MAX_FRAME = 1000

trc = []
tlc = []
blc = []
brc = []


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

#plt.plot(0, 0, colors[2] + "o")
#plt.plot(100, 100, colors[2] + "o")

for i in (blc, trc, tlc, brc):
    for j in i:

        if j[1] > 55:
            pass
        else:
            plt.plot(j[0], j[1], colors[count] + "o")
            data.append(j)

    count += 1

data = np.array(data)

clf = KMeans(n_clusters=4)
clf.fit(data)

for i in clf.cluster_centers_:
    print(i)
    plt.plot(i[0], i[1], "rx")

plt.savefig("plot.png")


tlcVal = clf.predict([tlc[0]])
blcVal = clf.predict([blc[0]])
trcVal = clf.predict([trc[0]])

print(tlcVal, trcVal, blcVal)


# this data is owned by Dmitry Kurtaev
# His page is https://github.com/dkurt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture("PerfectDataTRC.avi")

threshold = 60

guessedBLC = 0
guessedTLC = 0
guessedTRC = 0
guessedBRC = 0

Frames = 0
anotherCount = 0

while True:
    try:
        ret, img = cap.read()

        cv2.imshow("Improved Efficency", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (eyeX, eyeY, eyeW, eyeH) in eyes:
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

        cv2.imshow("ThE EYe YeS", eye_roi)

        k = cv2.waitKey(30) & 0xff

        if k == 97:
            raise ValueError

    except KeyboardInterrupt as erruir:

        print("Top left", guessedTLC, "\nTop right", guessedTRC, "\nBottom Left", guessedBLC,
              "\nBottom right", guessedBRC)

    except cv2.error as errir:

        print("Top left", guessedTLC, "\nTop right", guessedTRC, "\nBottom Left", guessedBLC,
              "\nBottom right", guessedBRC)


    except ValueError as e:

        print("Top left", guessedTLC, "\nTop right", guessedTRC, "\nBottom Left", guessedBLC,
              "\nBottom right", guessedBRC)

        file.close()
        break

    except NameError as n:
        pass


cap.release()
cv2.destroyAllWindows()
