"""LDA and KNN Classifier for Image Features
Ejercicio 2: Probar otras alteranativas para el reconocimiento"""


import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from ocr_classifier import OCRClassifier
from skimage.feature import hog


class NRKnnClassifier(OCRClassifier):
    def __init__(self, ocr_char_size, simple_preprocess=False):
        super().__init__(ocr_char_size)
        self.scaler = None
        self.knn = None
        self.simple_preprocess = simple_preprocess
    

    def preprocess(self, img):
        if self.simple_preprocess:
            # Preprocesamiento simple
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
            img = cv2.resize(img, (25, 25))
        else:
            # Preprocesamiento original (complejo)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (25, 25))
        
        return img.flatten()


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

        # Perform LDA training
        C = np.array(X).astype(np.float32)  # Convertir a float32
        E = np.array(y).astype(np.int32) # tipo openCV para etiquetas 
        self.scaler = StandardScaler()
        CR = self.scaler.fit_transform(C)  # Escalar características


        # Train classifier CR & E
        # First, we try with the normalBayesClassifier

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(CR, cv2.ml.ROW_SAMPLE, E)


        return C, E 
    

    def predict(self, img):
        features = self.preprocess(img) # Extract features from the image (e.g., pixel values, HOG, etc.)
        features_lda = self.scaler.transform([features]).astype(np.float32)  # Transformar usando LDA
        ret, result, neighbours, dist = self.knn.findNearest(features_lda, k=3)  # Predecir usando KNN
        return int(result[0][0])  # Devolver la etiqueta predicha como entero
