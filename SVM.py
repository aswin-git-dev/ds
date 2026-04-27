import cv2, os, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

path = 'Covid19-dataset'
classes = os.listdir(path)

img = cv2.imread('Covid19-dataset/Covid/01.jpeg', 0)
plt.imshow(img, cmap='gray'); plt.show()
img = cv2.resize(img, (64,64))
plt.imshow(img, cmap='gray'); plt.show()
img = cv2.Canny(img,100,200)
plt.imshow(img, cmap='gray'); plt.show()

x, y = [], []
for i,c in enumerate(classes):
    for f in os.listdir(os.path.join(path, c)):
        img = cv2.imread(os.path.join(path, c, f), 0)
        img = cv2.resize(img, (64, 64))
        img = cv2.Canny(img, 100, 200)
        x.append(img.flatten())
        y.append(i)

X = np.array(x)
y = np.array(y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

model = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)

ypred = model.predict(xtest)
print(classification_report(ytest, ypred))