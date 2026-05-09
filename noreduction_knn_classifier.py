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
        """
        Preprocesamiento avanzado: detecta polaridad, umbraliza, extrae el carácter
        y lo redimensiona MANTENIENDO la relación de aspecto y centrándolo.
        """
        # 1. Detectar polaridad de forma robusta mirando los bordes
        # El fondo siempre está en los bordes del recorte.
        h_img, w_img = img.shape[:2]
        border_mean = (np.mean(img[0, :]) + np.mean(img[-1, :]) + 
                       np.mean(img[:, 0]) + np.mean(img[:, -1])) / 4
        
        if border_mean > 127:
            # Fondo claro (letras negras - entrenamiento)
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            # Fondo oscuro (letras blancas - paneles)
            thresh_type = cv2.THRESH_BINARY

        # 2. Umbralización de Otsu para obtener el carácter en blanco (255)
        _, img_bin = cv2.threshold(img, 0, 255, thresh_type + cv2.THRESH_OTSU)

        # 3. Localizar el carácter (contornos)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(self.ocr_char_size[0] * self.ocr_char_size[1], dtype=np.float32)

        # Nos quedamos con el contorno más grande
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        char_crop = img_bin[y:y+h, x:x+w]
        
        # 4. Redimensionar manteniendo relación de aspecto
        target_w, target_h = self.ocr_char_size
        margin = 2
        max_size = target_w - 2 * margin
        
        f = max_size / max(w, h)
        new_w, new_h = int(w * f), int(h * f)
        new_w, new_h = max(1, new_w), max(1, new_h)
        
        char_resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 5. Centrar en un lienzo negro de forma robusta
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = char_resized
        
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
        self.scaler = StandardScaler()
        CR = self.scaler.fit_transform(C)  # Escalar características


        # Train classifier CR & E
        # First, we try with the normalBayesClassifier

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(CR, cv2.ml.ROW_SAMPLE, E)


        return C, E 
    

    def predict(self, img):
        features = self.preprocess(img) 
        # Escalar las características usando el scaler entrenado
        features_scaled = self.scaler.transform([features]).astype(np.float32)  
        ret, result, neighbours, dist = self.knn.findNearest(features_scaled, k=3) 
        return int(result[0][0])  # Devolver la etiqueta predicha como entero
