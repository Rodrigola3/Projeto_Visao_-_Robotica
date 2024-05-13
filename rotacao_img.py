import cv2
import matplotlib.pyplot as plt
import numpy as np
from posicao_cantos import *

# # # # # Caminho para a imagem
# caminho_img = "visao_Robotica\Fotos_Mesa_Sinuca\Mesa_Sinuca_Color_1.jpg"

# img_in_gray = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)


def rotaciona_img(img_in_gray):

    # Pontos Conhecidos da imagem distorcida em pixel (só mudar depois para a posição final da câmera)
    # x_inicial = zero_maquina(img_in_color)[0]
    # y_inicial = zero_maquina(img_in_color)[1]
    # x0 = 18
    # y0 = 12
    # x1 = 1896
    # y1 =  13
    # x2 = 0
    # y2 = 1010
    # x3 =  1920
    # y3 = 1027
    x0 = cantos(img_in_gray)[0][0]
    y0 = cantos(img_in_gray)[0][1]
    x1 = cantos(img_in_gray)[1][0]
    y1 = cantos(img_in_gray)[1][1]
    x2 = cantos(img_in_gray)[2][0]
    y2 = cantos(img_in_gray)[2][1]
    x3 = cantos(img_in_gray)[3][0]
    y3 = cantos(img_in_gray)[3][1]

    coef_p = 1920/0.69
    coef_p = 1920/0.69
  

    
    # Coeficiente que transforma de distância(m) para pixel
    

    # Pontos Conhecidos da imagem Restaurada
    u0 = 0*coef_p
    v0 = 0*coef_p
    u1 = 0.69*coef_p
    v1 = 0*coef_p
    u2 = 0*coef_p
    v2 = 0.37*coef_p
    u3 = 0.69*coef_p
    v3 = 0.37*coef_p

    # Conversão de imagem 
    TH = np.array([[x0, y0, 1, 0, 0, 0, -x0*u0, -y0*u0],
                [x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1],
                [x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2],
                [x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3],
                [0, 0, 0, x0, y0, 1, -x0*v0, -y0*v0],
                [0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1],
                [0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2],
                [0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3]], dtype=np.float64)

    # Inversa 
    TH_inv =  np.linalg.inv(TH)

    # Matriz u v
    matriz_uv = np.array([[u0],
                          [u1],
                          [u2],
                          [u3],
                          [v0],
                          [v1],
                          [v2],
                          [v3]], dtype=np.float64)

    # Tranformada resultante  ab (transformada responsável por sair da img distorcida para correta)
    matriz_ab = np.matmul(TH_inv, matriz_uv)

    # Tranformada resultate  em forma de array(matriz)
    TH_resultante = np.array([[matriz_ab[0], matriz_ab[1], matriz_ab[2]],
                              [matriz_ab[3], matriz_ab[4], matriz_ab[5]],
                              [matriz_ab[6], matriz_ab[7],            1]], dtype=np.float64)



    # # Definição do tamanho da matriz da imagem
    # img_cortada = img_in_gray[:, :]
    
    (h, w) = img_in_gray.shape 

    # Criando imagens com intensidade branca
    img_out = np.ones((h,w), dtype=np.uint8)*255

    # Preciso ir do p1 para o p0
    TH_inv_rev = np.linalg.inv(TH_resultante)

    # "For" responsável por fazer a tranformada
    for u in range(w):
        for v in range(h):


            p1= np.array( [ u, v, 1 ])
            p0 = np.matmul(TH_inv_rev, p1)
            
            x = int(p0[0]/p0[2]) # divido por w para garantir que seja 1 o último valor [x, y, w=1]
            y = int(p0[1]/p0[2])

            if (x>=0) and (x<w) and (y>=0) and (y<h):
                img_out[v,u] = img_in_gray[y,x]    # Y e V sao as linhas; X e U sao as colunas

    # Recorta a imagem do tamanho da mesa
    img_cortada_out = img_out[:1036, :] # y - x

    return img_cortada_out


# # # # Criação do plot
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))


# axs[0].imshow(img_in_gray, cmap='gray', vmin=0, vmax=255)
# axs[0].set_title('img_cortada')

# axs[1].imshow(rotaciona_img(img_in_gray), cmap='gray', vmin=0, vmax=255)
# axs[1].set_title('img rotacionada')
# plt.show()
    


