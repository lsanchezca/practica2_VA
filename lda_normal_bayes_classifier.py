# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminar la implementación
# Modificar como se crea conveniente (incluyendo métodos y parámetros), únicamente es una guía.

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None
    
    def preprocess(self, img):
        """
        Umbralize, adaptative threshold, contours, bounding Rect, etc. to extract features from the image.
        """
        # Adaptative thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


        # Contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # bounding Rect
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img = img[y:y+h, x:x+w]
        
        img = cv2.resize(img, (25,25))

        return img.flatten() # Return the feature vector (e.g., pixel values, HOG features, etc.) as a 1D array



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
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(C, E)
        CR = self.lda.transform(C).astype(np.float32)  # Convertir a float32 para OpenCV



        # Train classifier CR & E
        # First, we try with the normalBayesClassifier

        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(CR, cv2.ml.ROW_SAMPLE, E)


        return C, E 

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """

        features = self.preprocess(img) # Extract features from the image (e.g., pixel values, HOG, etc.)
        features = np.array([features], dtype=np.float32)
        CR = self.lda.transform(features)
        _,y = self.classifier.predict(CR) # Obtain the estimated label by the LDA + Bayes classifier

        return int(y.item())



