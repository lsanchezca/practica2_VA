"""
Este script solo se usa para hacer prints y pruebas, no es parte de la practica"
"""

import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
from string import digits, ascii_uppercase, ascii_lowercase
import numpy as np  


def load_char_images(char_path, target_size=(25, 25)):
    """
    Carga todas las imágenes de una carpeta de carácter.
    
    Args:
        char_path: ruta a la carpeta del carácter
        target_size: tamaño al que redimensionar las imágenes (ancho, alto)
    
    Returns:
        list: lista de imágenes redimensionadas en escala de grises
    """
    images = []
    
    if not os.path.exists(char_path):
        return images
    
    for img_file in os.listdir(char_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            img_path = os.path.join(char_path, img_file)
            try:
                # Leer la imagen en escala de grises
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Redimensionar al tamaño objetivo
                    img = cv2.resize(img, target_size)
                    images.append(img)
            except Exception as e:
                print(f"Error al cargar {img_path}: {e}")
    
    return images


def load_training_images_dict(train_path, target_size=(25, 25)):
    """
    Carga todas las imágenes de train_ocr en un diccionario.
    
    La estructura esperada es:
    - train_ocr/
        - 0/ (dígitos)
        - 1/
        - ...
        - 9/
        - may/ (mayúsculas)
            - A/
            - B/
            - ...
            - Z/
        - min/ (minúsculas)
            - a/
            - b/
            - ...
            - z/
    
    Args:
        train_path: ruta a la carpeta train_ocr
        target_size: tamaño al que redimensionar las imágenes (ancho, alto)
    
    Returns:
        dict: {carácter: [lista de imágenes]}
    """
    images_dict = {}
    
    # Cargar dígitos (0-9)
    for digit in digits:
        digit_path = os.path.join(train_path, digit)
        images = load_char_images(digit_path, target_size)
        if images:
            images_dict[digit] = images
            #print(f"  {digit}: {len(images)} imágenes")
    
    # Cargar mayúsculas (A-Z)
    may_path = os.path.join(train_path, 'may')
    for letter in ascii_uppercase:
        letter_path = os.path.join(may_path, letter)
        images = load_char_images(letter_path, target_size)
        if images:
            images_dict[letter] = images
            #print(f"  {letter}: {len(images)} imágenes")
    
    # Cargar minúsculas (a-z)
    min_path = os.path.join(train_path, 'min')
    for letter in ascii_lowercase:
        letter_path = os.path.join(min_path, letter)
        images = load_char_images(letter_path, target_size)
        if images:
            images_dict[letter] = images
            #print(f"  {letter}: {len(images)} imágenes")
    
    return images_dict


def print_dataset_info(images_dict):
    """Imprime información sobre el dataset cargado"""
    print("\n" + "="*50)
    print("INFORMACIÓN DEL DATASET")
    print("="*50)
    print(f"Total de clases: {len(images_dict)}")
    
    total_images = sum(len(imgs) for imgs in images_dict.values())
    print(f"Total de imágenes: {total_images}")
    
    print("\nDistribución por clase:")
    for char in sorted(images_dict.keys()):
        count = len(images_dict[char])
        print(f"  '{char}': {count} imágenes")
    print("="*50 + "\n")


def preprocess(img):
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



def train(images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        print(" has entrado en images_dict")

        # Take training images and do feature extraction
        
        X = [] # Feature vectors by rows
        y = [] # Labels for each row in X 

        for char, images in images_dict.items():
            for img in images:
                features = preprocess(img) # Extract features from the image (e.g., pixel values, HOG, etc.)
                X.append(features)
                y.append(char)

        # Perform LDA training
        C = np.array(X)
        y = np.array(y)
        print(C.shape, y.shape)
        # self.lda = LinearDiscriminantAnalysis()
        # self.lda.fit(C, y)
        # CR = self.lda.transform(C)


if __name__ == "__main__":
    # Uso del script
    train_path = "./train_ocr"
    
    # Cargar el diccionario de imágenes
    images_dict = load_training_images_dict(train_path)

    train(images_dict)

    
    # Ahora puedes usar images_dict en tu clasificador
    print("El diccionario 'images_dict' está listo para usar en:")
    print("  classifier.train(images_dict)")
