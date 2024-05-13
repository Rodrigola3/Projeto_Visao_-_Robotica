import cv2
import matplotlib.pyplot as plt

# Iniciar a câmera
camera = cv2.VideoCapture(1)
desired_width = 1920
desired_height = 1080
camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
rodando = True
primeiro_frame = True  # Flag para identificar o primeiro frame

# Enquanto rodando for verdadeiro (camera fica ligada)
while rodando:

    status, frame = camera.read()

    # Capturar e mostrar o primeiro frame usando matplotlib
    if primeiro_frame:

        # Converter o frame para escala de cinza
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame_gray.shape)
        # Mostrar o primeiro frame e sua versão em escala de cinza usando matplotlib
        # fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        # axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converter BGR para RGB
        # axs[0].set_title('Figura colorida')
        # axs[1].imshow(frame_gray, cmap='gray', vmin=0, vmax=255)
        # axs[1].set_title('Figura em Escala de Cinza')
        # plt.show()
        
        primeiro_frame = False  # Mudar a flag para não entrar mais aqui
        
        # Perguntar ao usuário se deseja salvar o frame
        salvar = input("Deseja salvar o primeiro frame? (sim/não): ").lower()
        if salvar == 'sim':
            # Salvar o primeiro frame em um arquivos
            cv2.imwrite('Mesa_Sinuca_Color_5.jpg', frame)
            # cv2.imwrite('Mesa_Sinuca_Gray_7s.jpg', frame_gray)
            print("Frame salvo com sucesso!")
    # Mostrar todos os frames na tela para continuar o vídeo
    cv2.imshow("Camera", frame)
    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xff == ord('q'):
        rodando = False

# Limpar recursos e fechar janelas
camera.release()
cv2.destroyAllWindows()
