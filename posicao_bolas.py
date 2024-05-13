import cv2
import matplotlib.pyplot as plt
import numpy as np

from rotacao_img import *
from filtro_bolas import *

# def posecicao_bolas():
# Função que Seleciona o Blob da imagem
def selectBlob(img_in, keypoints):
    num_labels, labels = cv2.connectedComponents(img_in)
    img_out = np.zeros_like(img_in, dtype = np.uint8)
    if len(keypoints) > 0:
        for KP in keypoints:
            line = int(KP.pt[0])
            column = int(KP.pt[1])
            selected_label = labels[column, line]
            img_label = np.where(labels == selected_label, 255, 0).astype('uint8')
            img_out = np.bitwise_or(img_out, img_label)
    return img_out

# Caminho para a imagem
caminho_img = "visao_Robotica\Fotos_Mesa_Sinuca\Mesa_Sinuca_Color_3.jpg"

# Carregando a imagem colorida 
img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)

# Aplica função filtro_bolas que retorna todas as bolas brancas já com a imagem cortada
img = filtro_bolas(img_in_color)

# Convertendo para escala de cinza para processamento
img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Configuração dos parâmetros do detector de blobs
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 1000
params.maxArea = 4500
params.filterByCircularity = True
params.minCircularity = 0.45
params.maxCircularity = 1.2
params.filterByConvexity = False
params.filterByInertia = False

# Criação do detector
detector = cv2.SimpleBlobDetector_create(params)

# Detecção dos blobs
KP = detector.detect(img_bin)
print(f'Total de Bolas: {len(KP)}')

# Armazenando as posições dos keypoints
posicoes = []
for c in range(len(KP)):
    Px, Py = int(KP[c].pt[0]), int(KP[c].pt[1])
    posicoes.append((Px, Py))
    # print(f' Bola {c+1} -> Posição x: {Px}, Posição y: {Py}')
print(posicoes)
# Seleciona os blobs detectados
img_blob = selectBlob(img_bin, KP)

# Convertendo o blob para imagem colorida para adicionar contornos e texto
img_color = cv2.cvtColor(img_blob, cv2.COLOR_GRAY2RGB)

# Encontrando contornos na imagem do blob
contours, hierarchy = cv2.findContours(img_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Desenhando contornos 
img_contorno = cv2.drawContours(img_color, contours, -1, (0,0,255), 2)

# Adicionando numeração dos cortornos Desenhados acima
for i, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img_contorno, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Plotando as imagens
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(img_in_color, cv2.COLOR_BGR2RGB))
axs[0].set_title('Figura colorida')
axs[1].imshow(img_bin, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('img_cortada')
axs[2].imshow(cv2.cvtColor(img_contorno, cv2.COLOR_BGR2RGB))
axs[2].set_title('img_contorno')
plt.show()
