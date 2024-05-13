import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho para a imagem
caminho_img = "visao_Robotica\Fotos_Mesa_Sinuca\Mesa_Sinuca_Color_3.jpg"

# Carregando a imagem colorida 
img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
if img_in_color is None:
    raise FileNotFoundError("A imagem não foi encontrada no caminho especificado.")
img_in_color_cropped = img_in_color[110:918, 112:1809]  # Recorte da imagem: y - x

# Convertendo a imagem para o espaço de cor HSV
img_HSV = cv2.cvtColor(img_in_color_cropped, cv2.COLOR_BGR2HSV)

# Dividindo a imagem HSV em seus canais individuais
H, S, V = cv2.split(img_HSV)

# Plota as imagens
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img_in_color_cropped, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(H, cmap='hsv')
plt.colorbar()
plt.title('Canal H (Matiz)')

plt.subplot(1, 4, 3)
plt.imshow(S, cmap='gray')
plt.colorbar()
plt.title('Canal S (Saturação)')

plt.subplot(1, 4, 4)
plt.imshow(V, cmap='gray')
plt.colorbar()
plt.title('Canal V (Valor)')

plt.show()
