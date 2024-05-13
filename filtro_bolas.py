import cv2
import numpy as np
import matplotlib.pyplot as plt


# Caminho para a imagem
caminho_img = "visao_Robotica\Fotos_Mesa_Sinuca\Mesa_Sinuca_Color_4.jpg"

# Carregando a imagem colorida 
img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)

img_in_color = img_in_color#[110:935, 112:1799]  # Recorte da imagem: y - x


def filtro_bolas(img_in_color):
    # Convertendo a imagem para o espaço de cor HSV
    img_HSV = cv2.cvtColor(img_in_color, cv2.COLOR_BGR2HSV)

    # Dividindo a imagem HSV em seus canais individuais
    H, S, V = cv2.split(img_HSV)

    # Criando a máscara que isola os pixels dentro do intervalo de verde
    # Ajustamos o limite para verde, baseado apenas no canal H
    mask = np.where((H >= 78) & (H <= 89), 0, 255).astype(np.uint8)

    # Resultado: colocar máscara branca em fundo preto
    result = cv2.bitwise_and(img_in_color, img_in_color, mask=mask)
    result[mask > 0] = (255, 255, 255)

    kernel_1 = np.ones((5,5), np.uint8)
    kernel_2 = np.ones((8,8), np.uint8)
    dilation = cv2.dilate(result, kernel_1, iterations=1) # esparrama
    erosion = cv2.erode(dilation, kernel_2, iterations = 2)
    return erosion[110:935, 112:1799], result[110:935, 112:1799]

# Plota as imagens

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_in_color, cv2.COLOR_BGR2RGB))
plt.title('Original')


plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(filtro_bolas(img_in_color)[0], cv2.COLOR_BGR2RGB))
plt.title('erosion')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(filtro_bolas(img_in_color)[1], cv2.COLOR_BGR2RGB))
plt.title('result')

plt.show()
