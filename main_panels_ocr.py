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
    classifier = NRKnnClassifier((25,25), simple_preprocess=True)
    classifier.train(training_images)


    # Load testing data
    if args.test_path:
        # Obtenemos lista de imágenes .png del directorio de test
        debug_lines_dir = os.path.join("outputs_classifier", "debug_lines")
        os.makedirs(debug_lines_dir, exist_ok=True)

        # Solo imágenes de entrada reales (evita procesar lines_*.png)
        test_images = [
            f for f in os.listdir(args.test_path)
            if f.endswith(".png") and not f.startswith("lines_")
        ]
        with open("resultado.txt", "w") as f_res:
            for img_name in test_images:
                img_path = os.path.join(args.test_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_blur = cv2.GaussianBlur(img, (3, 3), 0)
                # umbralizacion
                img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)

                contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # obtener cada digito
                H, W = img.shape[:2]
                char_boxes = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    area = w * h
                    ar = w / float(h)

                    if area < 20:
                        continue
                    if h < 8 or w < 2:
                        continue
                    if h > 0.45 * H or w > 0.45 * W:
                        continue
                    if ar < 0.08 or ar > 1.8:
                        continue

                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    char_boxes.append((x, y, w, h, cx, cy))


                # encontrar los caracteres alineados con RANSAC
                lines_idx = []
            remaining = np.arange(len(char_boxes))

            heights = np.array([b[3] for b in char_boxes], dtype=np.float32)
            med_h = float(np.median(heights)) if len(heights) else 12.0

            # RANSAC para líneas con >=3 caracteres
            if len(remaining) >= 3:
                ransac = linear_model.RANSACRegressor(
                    residual_threshold=max(3.0, 0.45 * med_h),
                    random_state=0
                )

                while len(remaining) >= 3:
                    pts = np.array([[char_boxes[i][4], char_boxes[i][5]] for i in remaining], dtype=np.float32)
                    X = pts[:, 0].reshape(-1, 1)
                    y_r = pts[:, 1]

                    ransac.fit(X, y_r)
                    inliers = ransac.inlier_mask_

                    if np.sum(inliers) < 3:
                        break

                    lines_idx.append(remaining[inliers].tolist())
                    remaining = remaining[~inliers]
            else:
                remaining = np.arange(len(char_boxes))

            # Fallback: agrupar por Y para líneas cortas (1-2 caracteres)
            if len(remaining) > 0:
                rem_sorted = sorted(remaining.tolist(), key=lambda i: char_boxes[i][5])
                clusters = []
                for idx in rem_sorted:
                    cy = char_boxes[idx][5]
                    placed = False
                    for cl in clusters:
                        mean_y = np.mean([char_boxes[j][5] for j in cl])
                        if abs(cy - mean_y) <= 0.8 * med_h:
                            cl.append(idx)
                            placed = True
                            break
                    if not placed:
                        clusters.append([idx])

                lines_idx.extend(clusters)

                # Dibujar las líneas detectadas
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for line_idx in lines_idx:
                    for i in line_idx:
                        x, y, w, h, cx, cy = char_boxes[i]
                        cv2.circle(img_color, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                        # Guardar la imagen con las líneas detectadas
                        output_path = os.path.join(debug_lines_dir, f"lines_{img_name}")
                        cv2.imwrite(output_path, img_color)
                
            

                # Ejecutar clasificador OCR sobre cada dígito detectado en orden

                # ordenar lineas por coordenada y (de arriba a abajo)
                # Orden de lectura: arriba->abajo y dentro de línea izquierda->derecha
                lines_idx.sort(key=lambda ln: np.mean([char_boxes[i][5] for i in ln]))

                line_texts = []
                for ln in lines_idx:
                    ln_sorted = sorted(ln, key=lambda i: char_boxes[i][4])
                    chars = []
                    for i in ln_sorted:
                        x, y, w, h, _, _ = char_boxes[i]

                        # IMPORTANTe: recorte real por bounding box, no recorte fijo por centro
                        char_img = img[y:y+h, x:x+w]

                        pred_label = classifier.predict(char_img)
                        pred_char = classifier.label2char(pred_label)
                        chars.append(pred_char)

                    if chars:
                        line_texts.append("".join(chars))

                # Formato pedido en el enunciado: líneas separadas por +
                output_text = "+".join(line_texts)
                                

                # Escribir en resultado.txt: nombre_fichero>;<x1>;<y1>;<x2>;<y2>;<tipo>;<score>;<texto_ocr>
                h, w = img.shape[:2]
                x1, y1 = 0, 0
                x2, y2 = w-1, h-1
                score = 1 # calcular  

                linea = f"{img_name};{int(x1)};{int(y1)};{int(x2)};{int(y2)};tipo?;{score:.2f};{output_text}\n"
                f_res.write(linea)

                with open("debug_panels.txt", "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"Panel: {img_name}\n")
                    f_debug.write(f"  Lineas detectadas: {len(lines_idx)}\n")

                    for i, line_idx in enumerate(lines_idx):
                        f_debug.write(f"    Linea {i+1}: {len(line_idx)} caracteres\n")

                    f_debug.write(f"  OCR final: {output_text}\n")
                    f_debug.write("-" * 40 + "\n")


                



            



            

















        
    


    # Evaluate OCR over road panels





