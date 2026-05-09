# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025


import argparse

# import panel_det
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI, guarda archivos en lugar de mostrar
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from string import digits, ascii_uppercase, ascii_lowercase
from ocr_classifier import OCRClassifier
from im_feature_PCA_KNN_classifier import PcaKnnClassifier
from noreduction_knn_classifier import NRKnnClassifier
import os 
import time

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
            

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
                    # Redimensionar al tamaño objetivo si se especifica
                    if target_size is not None:
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






if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default="", help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./test_ocr", help='Select the validation data dir')

    args = parser.parse_args()


    # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)

    print("1) LDA + Bayes con Gaussiana")
    print("2) PCA + KNN")
    print("3) No Reduction + KNN")
    choice = None

    choice = input("Introduce el número del clasificador a evaluar: ")

    if choice == "1":
        I = LdaNormalBayesClassifier((25,25))
    elif choice == "2":
        I = PcaKnnClassifier((25,25))
    elif choice == "3":
        I = NRKnnClassifier((25,25))
    

    train_images_dict = load_training_images_dict(args.train_path)

    
    # 2) Cargar datos de validación y sus etiquetas
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    classifier = OCRClassifier((25,25))
    val_images_dict = load_training_images_dict(args.validation_path) # Cargar imágenes de validación y etiquetas (ground trut)
    gt_labels = classifier.get_labels_dict(val_images_dict) # Obtener etiquetas de validación a partir del diccionario de imágenes de validación

    # 3) Entrenar clasificador
    train_start = time.perf_counter()
    C, E = I.train(train_images_dict)
    train_time = time.perf_counter() - train_start

    # 4) Ejecutar el clasificador sobre los datos de test
    predict_start = time.perf_counter()
    predicted_labels = I.predict_dict(val_images_dict)
    predict_time = time.perf_counter() - predict_start

        # 5) Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    precision_macro = sklearn.metrics.precision_score(gt_labels, predicted_labels, average='macro', zero_division=0)
    recall_macro = sklearn.metrics.recall_score(gt_labels, predicted_labels, average='macro', zero_division=0)
    f1_macro = sklearn.metrics.f1_score(gt_labels, predicted_labels, average='macro', zero_division=0)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(gt_labels, predicted_labels)

    print("\n================= RESUMEN DEL CLASIFICADOR =================")
    print(f"Clasificador evaluado: {choice}")
    print(f"Accuracy: {accuracy * 100:.2f} %")
    print(f"Precision macro: {precision_macro * 100:.2f} %")
    print(f"Recall macro: {recall_macro * 100:.2f} %")
    print(f"F1 macro: {f1_macro * 100:.2f} %")
    print(f"Balanced accuracy: {balanced_acc * 100:.2f} %")
    print(f"Tiempo de entrenamiento: {train_time:.4f} s")
    print(f"Tiempo total de predicción: {predict_time:.4f} s")
    print(f"Tiempo medio por muestra: {predict_time / max(len(predicted_labels), 1):.6f} s")

    original_dim = C.shape[1]
    if hasattr(I, 'pca') and I.pca is not None:
        reduced_dim = I.pca.n_components_
        compression = (1 - reduced_dim / original_dim) * 100
        print(f"Dimensionalidad original: {original_dim}")
        print(f"Dimensionalidad tras PCA: {reduced_dim}")
        print(f"Compresión: {compression:.2f} %")
    elif hasattr(I, 'lda') and I.lda is not None:
        reduced_dim = I.lda.coef_.shape[0]
        compression = (1 - reduced_dim / original_dim) * 100
        print(f"Dimensionalidad original: {original_dim}")
        print(f"Dimensionalidad tras LDA: {reduced_dim}")
        print(f"Compresión: {compression:.2f} %")
    else:
        print(f"Dimensionalidad de entrada: {original_dim}")
        print("Reducción de dimensionalidad: ninguna")

    print("============================================================\n")

    cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm)
    plt.savefig(f'confusion_matrix_{choice}.png')

    # ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt_labels, predicted_labels, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{choice}.png')
