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
    print(f"Cargando imágenes de entrenamiento desde: {args.train_path}...")
    training_images = load_training_images_dict(args.train_path, target_size=None)
    print(f"Imágenes cargadas ({sum(len(v) for v in training_images.values())} en total). Entrenando clasificador KNN...")

    # Create the OCR classifier
    # Usamos NRKnnClassifier que ha demostrado ser el más preciso en los paneles
    classifier = NRKnnClassifier((25, 25), simple_preprocess=False)

    print(f"Entrenando clasificador KNN...")
    classifier.train(training_images)
    print(f"Entrenamiento completado. Procesando paneles de test en: {args.test_path}...")


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
            for i, img_name in enumerate(test_images):
                print(f"[{i+1}/{len(test_images)}] Procesando: {img_name}")
                img_path = os.path.join(args.test_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_blur = cv2.GaussianBlur(img, (3, 3), 0)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_blur)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
                img_tophat = cv2.morphologyEx(img_clahe, cv2.MORPH_TOPHAT, kernel)
                
                _, img_otsu = cv2.threshold(img_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                if cv2.countNonZero(img_otsu) < 100:
                    img_thresh = cv2.adaptiveThreshold(img_tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
                else:
                    img_thresh = img_otsu
                
                # Limpieza morfológica 3x3 para manejar el ruido del CLAHE
                clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, clean_kernel)

                # RETR_LIST para ver dentro de recuadros, pero con filtros más firmes para evitar ruido
                contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # obtener cada digito
                H, W = img.shape[:2]
                char_boxes = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    ar = w / float(h)
                    
                    # Filtros equilibrados: permiten el '1' pero bloquean ruido pequeño
                    if h < 12 or w < 3:
                        continue
                    if h > 0.85 * H or w > 0.85 * W:
                        continue
                    if ar < 0.12 or ar > 3.0: 
                        continue

                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    char_boxes.append((x, y, w, h, cx, cy))

                # 1. Eliminar cajas anidadas (agujeros de letras por RETR_LIST)
                char_boxes.sort(key=lambda b: b[2]*b[3], reverse=True) # De mayor a menor área
                filtered_boxes = []
                for box in char_boxes:
                    is_nested = False
                    for fbox in filtered_boxes:
                        # Si la caja 'box' está dentro de 'fbox', la marcamos como anidada
                        if box[0] >= fbox[0]-1 and box[1] >= fbox[1]-1 and \
                           box[0]+box[2] <= fbox[0]+fbox[2]+1 and \
                           box[1]+box[3] <= fbox[1]+fbox[3]+1:
                            is_nested = True
                            break
                    if not is_nested:
                        filtered_boxes.append(box)
                char_boxes = filtered_boxes


                # encontrar los caracteres alineados con RANSAC
                lines_idx = []
                remaining = np.arange(len(char_boxes))

                heights = np.array([b[3] for b in char_boxes], dtype=np.float32)
                med_h = float(np.median(heights)) if len(heights) else 12.0

                # RANSAC para líneas con >=3 caracteres
                if len(remaining) >= 3:
                    ransac = linear_model.RANSACRegressor(
                        residual_threshold=max(2.0, 0.2 * med_h),
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

                # --- Lógica de visualización (Figura 3) ---
                debug_dir = "debug_visualizations"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 1. Imagen Binaria
                v1 = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
                
                # 2. Detecciones y RANSAC
                v2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for x, y, w, h, _, _ in char_boxes:
                    cv2.rectangle(v2, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # 3. Resultado OCR Final
                v3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Ordenar líneas de arriba a abajo
                lines_idx.sort(key=lambda ln: np.mean([char_boxes[i][5] for i in ln]))
                
                line_texts = []
                for ln in lines_idx:
                    ln_sorted = sorted(ln, key=lambda i: char_boxes[i][4])
                    
                    # Dibujar línea RANSAC en v2
                    if len(ln_sorted) >= 2:
                        p1 = (int(char_boxes[ln_sorted[0]][4]), int(char_boxes[ln_sorted[0]][5]))
                        p2 = (int(char_boxes[ln_sorted[-1]][4]), int(char_boxes[ln_sorted[-1]][5]))
                        cv2.line(v2, p1, p2, (255, 255, 0), 1)

                    chars = []
                    for i in ln_sorted:
                        x, y, w, h, _, _ = char_boxes[i]
                        char_img = img[y:y+h, x:x+w]
                        pred_label = classifier.predict(char_img)
                        pred_char = classifier.label2char(pred_label)
                        chars.append(pred_char)
                        
                        # Dibujar en v3
                        cv2.rectangle(v3, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        cv2.putText(v3, pred_char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    if chars:
                        line_texts.append("".join(chars))

                output_text = "+".join(line_texts)
                
                # Crear mosaico comparativo
                mosaico = np.hstack((v1, v2, v3))
                cv2.imwrite(os.path.join(debug_dir, f"debug_{img_name}"), mosaico)

                # Escribir en resultado.txt
                linea = f"{img_name};0;0;{img.shape[1]-1};{img.shape[0]-1};tipo?;1.00;{output_text}\n"
                f_res.write(linea)


                



            



            

















        
    


    # Evaluate OCR over road panels





