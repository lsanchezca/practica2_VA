import argparse
from ocr_classifier import OCRClassifier
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI, guarda archivos en lugar de mostrar
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from string import digits, ascii_uppercase, ascii_lowercase
from im_feature_PCA_KNN_classifier import PcaKnnClassifier
from noreduction_knn_classifier import NRKnnClassifier
from evaluar_clasificadores_OCR import load_training_images_dict, plot_confusion_matrix
from sklearn import linear_model

import os 
import time



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="./test_ocr_panels", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data
    training_images = load_training_images_dict(args.train_path, target_size=(25, 25))



    # Create the OCR classifier
    classifier = NRKnnClassifier((25,25))
    classifier.train(training_images)


    # Load testing data
    if args.test_path:
        # Obtenemos lista de imágenes .png del directorio de test
        test_images = [f for f in os.listdir(args.test_path) if f.endswith('.png')]
        with open("resultado.txt", "w") as f_res:
            for img_name in test_images:
                img_path = os.path.join(args.test_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # umbralizacion
                _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
                # Redimensionar a 25x25
                img_resized = cv2.resize(img_thresh, (25, 25), interpolation=cv2.INTER_AREA)
                contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # obtener cada digito
                centers = []
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    digit = img_thresh[y:y+h, x:x+w]
                    cx = x + w/2
                    cy = y + h/2
                    centers.append((cx, cy))

                
                # encontrar los caracteres alineados con RANSAC
                lines = []
                remaining = np.array(centers)

                ransac = linear_model.RANSACRegressor()

                while len(remaining) > 2:

                    X = remaining[:,0].reshape(-1,1)   # coordenadas x
                    y = remaining[:,1]                 # coordenadas y

                    ransac.fit(X, y)

                    inliers = ransac.inlier_mask_
                    line_points = remaining[inliers]   # centros de la línea detectada

                    lines.append(line_points)

                        # quitar los puntos ya usados
                    remaining = remaining[~inliers]
                

                # Dibujar las líneas detectadas
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for line in lines:
                    for (cx, cy) in line:
                        cv2.circle(img_color, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                        # Guardar la imagen con las líneas detectadas
                        output_path = os.path.join(args.test_path, f"lines_{img_name}")
                        cv2.imwrite(output_path, img_color)
                
            

                # Ejecutar clasificador OCR sobre cada dígito detectado en orden

                # ordenar lineas por coordenada y (de arriba a abajo)
                lines.sort(key=lambda line: np.mean(line[:,1]))

                ordered_lines = []
                for line in lines:
                    line_sorted = sorted(line, key=lambda p: p[0])   # ordenar por X
                    ordered_lines.append(line_sorted)
                
                output_text = ""
                for line in ordered_lines:
                    if output_text != "":
                        output_text += "+"
                    for (cx, cy) in line:
                        # Extraer el dígito de la imagen original usando un recorte alrededor del centro
                        x1 = int(cx - 12)
                        y1 = int(cy - 12)
                        x2 = int(cx + 12)
                        y2 = int(cy + 12)

                        digit_img = img_thresh[y1:y2, x1:x2]

                        # Preprocesar el dígito para el clasificador
                        digit_img_resized = cv2.resize(digit_img, (25, 25), interpolation=cv2.INTER_AREA)

                        # Clasificar el dígito usando el OCR
                        predicted_char = classifier.predict(digit_img_resized)
                        output_text += str(predicted_char)
                

                # Escribir en resultado.txt: nombre_fichero>;<x1>;<y1>;<x2>;<y2>;<tipo>;<score>;<texto_ocr>
                h, w = img.shape[:2]
                x1, y1 = 0, 0
                x2, y2 = w-1, h-1
                score = 1 # calcular  

                linea = f"{img_name};{int(x1)};{int(y1)};{int(x2)};{int(y2)};tipo?;{score:.2f};{output_text}\n"
                f_res.write(linea)


                



            



            

















        
    


    # Evaluate OCR over road panels





