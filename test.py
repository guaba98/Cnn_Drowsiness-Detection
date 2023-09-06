import numpy as np
import pandas as pd
import os
import cv2

# 특정 폴더 내에 있는 폴더 리스트화
labels = os.listdir(r"image\train")
# print(labels) # ['Closed', 'no_yawn', 'Open', 'yawn']

# 사진 테스트로 띄우기
import matplotlib.pyplot as plt
plt.imshow(plt.imread(r"image\train\Closed\_10.jpg"))
# plt.show()

# 이미지 쉐입
a = plt.imread(r"image\train\yawn\10.jpg")
# print(a.shape) # (480, 640, 3)

# 이미지 띄우기
plt.imshow(plt.imread(r"image\train\yawn\10.jpg"))
# plt.show()

# 얼굴인식하여 이미지 크롭하여 리스트에 저장
def face_for_yawn(direc="image/train", face_cas_path="haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = img[y:y + h, x:x + w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


# yawn_no_yawn = face_for_yawn()
# print(yawn_no_yawn)

# 감은 눈과 안 감은 눈
def get_data(dir_path = "image/train", eye_cas="haarcascade_frontalface_default.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num += 2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data


data_train = get_data()


#이미지 확장과 어레이 변환
def append_data():
    # total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)

# 저장할 새 변수
new_data = append_data()

# 라벨로 변환
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

# 어레이 재변환
X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

# 라벨 바이너리화
from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)

# 라벨 어레이
y = np.array(y)

# 훈련자료/테스트자료 분리
from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

# X_test 길이
print(len(X_test))

# 딥러닝 모델 임포트
f