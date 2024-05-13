import cv2
import matplotlib.pyplot as plt
import numpy as np

from rotacao_img import *

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
caminho_img = "visao_Robotica\Fotos_Mesa_Sinuca\Mesa_Sinuca_Color_4.jpg"

# Carregando a imagem colorida 
img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)

# Carregando a imagem em escala de cinza
img_in_gray = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

# Img rotacionada e cortada
img_cortada = rotaciona_img(img_in_gray)#[22:1006,20:1875]

# Filtro para identificar a bola branca
img_bin= (np.where(img_cortada > 180 , 255, 0 )).astype(np.uint8)

# Parâmestros gerais para identificar o Blob 
params = cv2.SimpleBlobDetector_Params()

# Set blob color (0=black, 255=white)
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 3000
params.maxArea = 5000
params.filterByCircularity = True
params.filterByConvexity = False
params.filterByInertia=False

# Detector de Parametros definidos acima
detector = cv2.SimpleBlobDetector_create(params)

# # Detect blobs para a imagem 
KP = detector.detect(img_bin)  # pega o centro do blob
print(len(KP))

# Coeficiente que transforma de pixel para distância em metros
coef_d = 0.69/1920

# Pega o Px e Py da bola branca
Px, Py = int(KP[0].pt[0])*coef_d, int(KP[0].pt[1])*coef_d
print(f'Posição x: {Px}m, Posição y: {Py}m')

# Aplica a função selectBlob_Amassado na imagem binarizada acima
img_blob= selectBlob(img_bin, KP)

# Tranforma img_blob para color para poder colocar a borda Vermelha
img_color= cv2.cvtColor(img_blob, cv2.COLOR_GRAY2RGB)

# Retorna os cortornos 
contours, hierarchy = cv2.findContours(img_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# "Desenho" o contorno na imagem
img_contorno = cv2.drawContours(img_color, contours, -1, (255,0,0), 2)


# Plota as imagens 
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(img_in_color, cv2.COLOR_BGR2RGB))  # Converter BGR para RGB
axs[0].set_title('Figura colorida')

axs[1].imshow(img_cortada, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('img_cortada')

axs[2].imshow(img_bin, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('img_bin')


axs[2].imshow(img_contorno)
axs[2].set_title('img_contorno')
plt.show()
        

