import cv2
import matplotlib.pyplot as plt
import numpy as np


# Caminho para a imagem
caminho_img = "visao_Robotica/Fotos_Mesa_Sinuca/Mesa_Sinuca_Color_1.jpg"

# Carregando a imagem colorida 
img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
# Recorta a imagem do tamanho da mesa
# img_in_color = img_in_color[1:1028, :] # y - x
# Carregando a imagem em escala de cinza
img_in_gray = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
# Filtro para identificar a bola branca
img_bin = (np.where(img_in_gray > 220, 255, 0)).astype(np.uint8)


kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(img_bin, kernel, iterations=3)
erosion = cv2.erode(dilation, kernel, iterations = 3)



# Plota as imagens
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Exibir a imagem colorida
axs[0].imshow(cv2.cvtColor(img_in_color, cv2.COLOR_BGR2RGB))  # Converter BGR para RGB
axs[0].set_title('img_in_color')

# Exibir a imagem em escala de cinza
axs[1].imshow(img_in_gray, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('img_in_gray')

plt.show()
