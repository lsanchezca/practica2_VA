"""PCA and KNN Classifier for Image Features
Ejercicio 2: Probar otras alteranativas para el reconocimiento"""


import cv2
import numpy as np
from sklearn.decomposition import PCA
from ocr_classifier import OCRClassifier
from skimage.feature import hog


class PcaKnnClassifier(OCRClassifier):
    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.pca = None
        self.knn = None
    

    def preprocess(self, img):
        # Adaptative thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


        # Contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # bounding Rect
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img = img[y:y+h, x:x+w]
        
        img = hog(img, pixels_per_cell=(8,8), cells_per_block=(3,3), feature_vector=True) # Extract HOG features
        return img


    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Take training images and do feature extraction
        
        X = [] # Feature vectors by rows
        y = [] # Labels for each row in X 

        for char, images in images_dict.items():
            for img in images:
                features = self.preprocess(img) # Extract features from the image (e.g., pixel values, HOG, etc.)
                X.append(features)
                y.append(self.char2label(char))

        # Perform PCA   training
        C = np.array(X).astype(np.float32)  # Convertir a float32
        E = np.array(y).astype(np.int32) # tipo openCV para etiquetas 
        self.pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza
        self.pca.fit(C)
        CR = self.pca.transform(C).astype(np.float32)  # Convertir a float32 para OpenCV



        # Train classifier CR & E
        # First, we try with the normalBayesClassifier

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(CR, cv2.ml.ROW_SAMPLE, E)


        return C, E 
    

    def predict(self, img):
        features = self.preprocess(img) # Extract features from the image (e.g., pixel values, HOG, etc.)
        features_pca = self.pca.transform([features]).astype(np.float32)  # Transformar usando PCA
        ret, result, neighbours, dist = self.knn.findNearest(features_pca, k=3)  # Predecir usando KNN
        return int(result[0][0])  # Devolver la etiqueta predicha como entero
