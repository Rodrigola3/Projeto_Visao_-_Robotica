import cv2
import matplotlib.pyplot as plt
import numpy as np


# # Caminho para a imagem
# caminho_img = "visao_Robotica\\Fotos_Mesa_Sinuca\\Mesa_Sinuca_Color_1.jpg"

# # Carregando a imagem colorida e em escala de cinza
# img_in_color = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
# img_in_gray = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

# # Função que seleciona o Blob da imagem
# def selectBlob(img_in, keypoints):
#     num_labels, labels = cv2.connectedComponents(img_in)
#     img_out = np.zeros_like(img_in, dtype=np.uint8)
#     if len(keypoints) > 0:
#         for KP in keypoints:
#             x = int(KP.pt[0])
#             y = int(KP.pt[1])
#             selected_label = labels[y, x]
#             img_label = np.where(labels == selected_label, 255, 0).astype('uint8')
#             img_out = np.bitwise_or(img_out, img_label)
#     return img_out

def cantos(img_in_gray):
    # Cortando a parte útil da imagem em escala de cinza
    img_cortada = img_in_gray[1:1028, :]

    # Filtro para identificar a bola branca
    img_bin = (np.where(img_cortada > 220, 255, 0)).astype(np.uint8)

    # # Parâmetros para identificar o Blob
    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByColor = True
    # params.blobColor = 255
    # params.filterByArea = True
    # params.minArea = 5000
    # params.maxArea = 11000
    # params.filterByCircularity = False
    # params.filterByConvexity = False
    # params.filterByInertia = False

    # Detector de Blob baseado nos parâmetros definidos
    # detector = cv2.SimpleBlobDetector_create(params)

    # Detecta blobs na imagem
    # keypoints = detector.detect(img_bin)
    # print(f'Blobs detectados: {len(keypoints)}')

    # Aplica a função de seleção de Blob na imagem binarizada
    # img_blob = selectBlob(img_bin, keypoints)

    # # Transforma a imagem do blob para colorida para adicionar borda vermelha
    # img_color = cv2.cvtColor(img_blob, cv2.COLOR_GRAY2RGB)

    # # Encontra contornos na imagem
    # contours, hierarchy = cv2.findContours(img_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # Desenha os contornos na imagem
    # img_contorno = cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)

    # Encontrando posições brancas nas quatro regiões:
    posicoes = {
        'superior esquerda': None,
        'inferior esquerda': None,
        'superior direita': None,
        'inferior direita': None
    }
    # Dimensões da imagem cortada
    h, w = img_cortada.shape
    # print(f'Dimensões da imagem cortada: altura={h}, largura={w}')

    # Loop unificado para busca
    lista_posicoes = []
    for y in range(h):
        for x in range(w):
            if img_bin[y, x] == 255:
                if y < 40 and x < 40 and not posicoes['superior esquerda']:
                    posicoes['superior esquerda'] = (x, y)
                    lista_posicoes.append((x, y))
                if y >= 1010 and x < 40 and not posicoes['inferior esquerda']:
                    posicoes['inferior esquerda'] = (x, y)
                    lista_posicoes.append((x, y))
                if y < 40 and x > 1895 and not posicoes['superior direita']:
                    posicoes['superior direita'] = (x, y)
                    lista_posicoes.append((x, y))
                if y >= 1010 and x > 1895 and not posicoes['inferior direita']:
                    posicoes['inferior direita'] = (x, y)
                    lista_posicoes.append((x, y))

    # # Impressão formatada das posições
    # for key, value in posicoes.items():
    #     print(f'Primeira posição branca na parte {key}: {value}')
    return lista_posicoes

# print(cantos(img_in_gray)[0][0])

# # Plota as imagens
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# axs[0].imshow(cv2.cvtColor(img_in_color, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Imagem de Entrada')
# axs[1].imshow(cv2.cvtColor(img_blob, cv2.COLOR_GRAY2RGB))
# axs[1].set_title('Imagem de Saída')
# plt.show()
