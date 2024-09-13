
import math

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import Process

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

img = cv2.imread("./bienso/10.jpg")
img = cv2.resize(img, dsize=(1080, 1080))

######## Upload KNN model ######################
npaClassifications = np.loadtxt("./classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("./flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
#########################
# CNN Model
# npaFlattenedImages = np.loadtxt("./flattened_images.txt", np.float32)
# n_samples = npaFlattenedImages.shape[0]
# npaFlattenedImages = npaFlattenedImages.reshape(n_samples, 30, 20, 1)  # 30x20 là kích thước ảnh và 1 là số kênh (trắng đen)

# npaClassifications = np.loadtxt("./classifications.txt", np.float32)
# npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
# def convert_labels(label):
#     if 48 <= label <= 57:  # '0' to '9'
#         return label - 48
#     elif 65 <= label <= 90:  # 'A' to 'Z'
#         return label - 55
#     else:
#         raise ValueError("Invalid label value: {}".format(label))

# # Áp dụng hàm cho tất cả các nhãn
# npaClassifications = np.vectorize(convert_labels)(npaClassifications)

# # Khởi tạo mô hình CNN
# model = models.Sequential()

# # Thêm lớp tích chập (Convolutional Layer) và lớp pooling
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 20, 1)))  # Lớp tích chập đầu vào
# model.add(layers.MaxPooling2D((2, 2)))

# # Thêm lớp tích chập tiếp theo
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

# # Thêm lớp fully connected (kết nối đầy đủ)
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))

# # Lớp đầu ra với số lớp bằng số ký tự cần phân loại
# model.add(layers.Dense(36, activation='softmax'))  # 36 là số ký tự (0-9, A-Z)

# # Biên dịch mô hình
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Huấn luyện mô hình
# model.fit(npaFlattenedImages, npaClassifications, epochs=10, batch_size=64, validation_split=0.2)



################ Image Preprocessing #################
imgGrayscaleplate, imgThreshplate = Process.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
# cv2.imshow("dilated_image",dilated_image)

###########################################

###### Draw contour and filter out the license plate  #############
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
# cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các ctour trong hình lớn

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    if (len(approx) == 4):
        screenCnt.append(approx)

        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Find the angle of the license plate #####################
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        ####################################

        ########## Crop out the license plate and align it to the right angle ################

        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

        # Cropping
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        ####################################

        #################### Prepocessing and Character segmentation ####################
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

        ##################### Filter out characters #################
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind


        ############ Character recognition ##########################

        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""
        ######KNN######
        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
            strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            if (y < height / 3):  # decide 1 or 2-line license plate
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))


        ########CNN#####
        # for i in char_x:
        #     (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #     imgROI = thre_mor[y:y + h, x:x + w]  # Crop ký tự
        #     imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize ảnh
        #     imgROIResized = imgROIResized.reshape(1, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, 1)  # Định dạng lại cho CNN

        #     imgROIResized = np.float32(imgROIResized) / 255.0  # Chuẩn hóa dữ liệu về 0-1

        #     # Dự đoán ký tự bằng CNN
        #     prediction = model.predict(imgROIResized)
        #     strCurrentChar = chr(np.argmax(prediction))  # ASCII của ký tự dự đoán

        #     cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

        #     if y < height / 3:
        #         first_line += strCurrentChar
        #     else:
        #         second_line += strCurrentChar

        # print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        # cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        n = n + 1

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img)

cv2.waitKey(0)
