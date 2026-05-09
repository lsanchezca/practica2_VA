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
        Preprocesamiento avanzado: detecta polaridad, umbraliza con Otsu, 
        extrae el carácter y lo redimensiona centrándolo.
        """
        # 1. Detectar polaridad de forma robusta mirando los bordes
        h_img, w_img = img.shape[:2]
        border_mean = (np.mean(img[0, :]) + np.mean(img[-1, :]) + 
                       np.mean(img[:, 0]) + np.mean(img[:, -1])) / 4
        
        if border_mean > 127:
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            thresh_type = cv2.THRESH_BINARY

        # 2. Umbralización de Otsu
        _, img_bin = cv2.threshold(img, 0, 255, thresh_type + cv2.THRESH_OTSU)

        # 3. Localizar el carácter (contornos)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(self.ocr_char_size[0] * self.ocr_char_size[1], dtype=np.float32)

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        char_crop = img_bin[y:y+h, x:x+w]
        
        # 4. Redimensionar manteniendo relación de aspecto y centrar
        target_w, target_h = self.ocr_char_size
        margin = 2
        max_size = target_w - 2 * margin
        f = max_size / max(w, h)
        new_w, new_h = max(1, int(w * f)), max(1, int(h * f))
        char_resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        canvas[(target_h-new_h)//2 : (target_h-new_h)//2 + new_h, 
               (target_w-new_w)//2 : (target_w-new_w)//2 + new_w] = char_resized
        
        return canvas.flatten().astype(np.float32)



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



