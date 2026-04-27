import cv2, os, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

path = 'Covid19-dataset'
classes = os.listdir(path)

img = cv2.imread('Covid19-dataset/Covid/01.jpeg', 0)
plt.imshow(img, cmap='gray'); plt.show()
img = cv2.resize(img, (64,64))
plt.imshow(img, cmap='gray'); plt.show()
img = cv2.Canny(img,100,200)
plt.imshow(img, cmap='gray'); plt.show()

X, y = [], []
for i,c in enumerate(classes):
    for f in os.listdir(os.path.join(path, c)):
        img = cv2.imread(os.path.join(path, c, f), 0)
        img = cv2.resize(img, (64, 64))
        img = cv2.Canny(img, 100, 200)
        X.append(img)
        y.append(i)
    
X = np.array(X)[...,None]
y = np.eye(len(classes))[y]

Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2)

model = models.Sequential([
    layers.Conv2D(8,3,activation='relu',input_shape=(64,64,1)),
    layers.Flatten(),
    layers.Dense(len(classes),activation='softmax')
])

model.compile('adam','categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtr,Ytr,epochs=5)
ypred = np.argmax(model.predict(Xte),1) 
print(classification_report(np.argmax(Yte,1), ypred))