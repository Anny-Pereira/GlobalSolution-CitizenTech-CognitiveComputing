"""
ETAPA 1 - PRÉ-PROCESSAMENTO
===========================
Objetivo: preparar a imagem e separar a fruta/verdura do fundo.

Conteúdo da aula usado aqui:
  - Lab01/Lab02: leitura de imagem e segmentação por pixel
  - Lab04: filtros de convolução (GaussianBlur para tirar ruído)
  - Lab05: espaço de cor (BGR -> HSV) e contornos
  - Lab06: morfologia matemática (abertura e fechamento)
"""

import cv2
import numpy as np

# Todas as imagens ficam do mesmo tamanho. Isso deixa as características
# comparáveis entre imagens e deixa o processamento mais rápido.
TAMANHO = (200, 200)


def carregar_imagem(caminho):
    """Lê a imagem do disco. O OpenCV lê as cores na ordem BGR (não RGB)."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {caminho}")
    return imagem


def redimensionar(imagem):
    """Coloca a imagem no tamanho padrão definido em TAMANHO."""
    return cv2.resize(imagem, TAMANHO)


def segmentar_fruta(imagem):
    """
    Cria uma MÁSCARA: uma imagem preto e branco onde
      branco (255) = fruta  |  preto (0) = fundo.

    Ideia: a fruta costuma ser bem mais "colorida" (saturada) do que o fundo
    (mesa, bancada, fundo branco/cinza). Por isso olhamos o canal S do HSV.

    Retorna: (mascara, contorno_da_fruta)
    """
    # Passo 1: converte de BGR para HSV.
    #   H = matiz (qual é a cor), S = saturação (quão viva é a cor),
    #   V = valor (quão clara/escura é a cor).
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    saturacao = hsv[:, :, 1]

    # Passo 2: desfoque gaussiano para suavizar ruídos antes de limiarizar.
    saturacao = cv2.GaussianBlur(saturacao, (5, 5), 0)

    # Passo 3: limiarização de Otsu. O Otsu escolhe sozinho o melhor limiar
    # que separa "pouco saturado" (fundo) de "muito saturado" (fruta).
    _, mascara = cv2.threshold(saturacao, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Passo 4: morfologia.
    #   Abertura -> remove pontinhos brancos soltos (ruído no fundo).
    #   Fechamento -> fecha buracos pretos dentro da fruta (ex.: manchas).
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Passo 5: contornos. Ficamos só com o MAIOR contorno, que deve ser a fruta.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    area_total = imagem.shape[0] * imagem.shape[1]
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        # Se o maior contorno for muito pequeno (< 5% da imagem), a
        # segmentação provavelmente falhou.
        area = cv2.contourArea(maior)
        # Se a "fruta" ocupa quase a foto toda, a foto não tem fundo
        # (ex.: várias maçãs empilhadas). Aí usamos o Plano B (abaixo).
        if 0.05 * area_total < area < 0.60 * area_total:
            mascara_final = np.zeros_like(mascara)
            # Pinta o contorno preenchido: assim manchas escuras DENTRO da
            # fruta continuam fazendo parte dela (são importantes p/ "podre").
            cv2.drawContours(mascara_final, [maior], -1, 255,
                             thickness=cv2.FILLED)
            return mascara_final, maior

    # Plano B: não deu para separar a fruta do fundo (foto sem fundo, com
    # várias frutas, ou fundo colorido). Então olhamos só um CÍRCULO NO
    # CENTRO da foto (região de interesse), onde normalmente está a fruta
    # principal. Assim evitamos contar as sombras entre as frutas.
    altura, largura = imagem.shape[:2]
    mascara_final = np.zeros((altura, largura), dtype=np.uint8)
    centro = (largura // 2, altura // 2)
    raio = int(min(altura, largura) * 0.25)
    cv2.circle(mascara_final, centro, raio, 255, thickness=cv2.FILLED)
    contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    return mascara_final, contornos[0]


def preprocessar(caminho_ou_imagem):
    """
    Junta tudo: carrega (se for um caminho), redimensiona e segmenta.
    Aceita tanto o caminho de um arquivo quanto uma imagem já carregada
    (útil para a webcam).
    """
    if isinstance(caminho_ou_imagem, np.ndarray):
        imagem = caminho_ou_imagem
    else:
        imagem = carregar_imagem(caminho_ou_imagem)
    imagem = redimensionar(imagem)
    mascara, contorno = segmentar_fruta(imagem)
    return imagem, mascara, contorno
