"""
ETAPA 2 - MEDIR A FRUTA
=======================
Depois de separar a fruta do fundo, medimos duas coisas:

  1) A "impressão digital de cor" da fruta (histograma de cor)
     -> serve para descobrir QUAL é a fruta.
  2) Quanto da fruta está manchado (escuro, marrom ou mofo)
     -> serve para descobrir se ela está PODRE.

Conteúdo da aula usado aqui:
  - Lab03: histograma (cv2.calcHist, cv2.compareHist)
  - Lab02 e Lab05: segmentação por cor em HSV (cv2.inRange)
"""

import cv2
import numpy as np


def converter_hsv(imagem):
    """BGR -> HSV. H = cor, S = quão viva é a cor, V = quão clara."""
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def histograma_cor(hsv, mascara):
    """
    Conta quantos pixels da fruta têm cada cor (canais H e S).
    É como uma "impressão digital de cor": banana tem muito amarelo,
    maçã muito vermelho, uva muito roxo.
    Só olha os pixels da fruta (mascara), não o fundo.
    """
    hist = cv2.calcHist([hsv], [0, 1], mascara, [30, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)  # deixa comparável, não importa o tamanho da fruta
    return hist


def semelhanca(hist1, hist2):
    """
    Compara duas impressões digitais de cor (cv2.compareHist).
    Resultado perto de 1 = cores muito parecidas. Perto de 0 = diferentes.
    """
    return max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


# Faixas de cor (HSV) que indicam fruta estragando. São um ponto de partida:
# o notebook mostra as máscaras para vocês conferirem e ajustarem.
#            nome            (H, S, V) mínimo     (H, S, V) máximo
FAIXAS_MANCHA = {
    "escuro":        ((0, 0, 0),       (180, 255, 50)),   # partes pretas
    "marrom":        ((5, 50, 30),     (25, 255, 150)),   # partes marrons
    "mofo":          ((0, 0, 160),     (180, 50, 255)),   # partes brancas/cinzas
}


def mascara_manchas(hsv, mascara):
    """Junta as 3 faixas de mancha em uma só máscara (só dentro da fruta)."""
    manchas = np.zeros(mascara.shape, dtype=np.uint8)
    for minimo, maximo in FAIXAS_MANCHA.values():
        faixa = cv2.inRange(hsv, np.array(minimo), np.array(maximo))
        manchas = cv2.bitwise_or(manchas, faixa)
    return cv2.bitwise_and(manchas, mascara)


def percentual_manchas(hsv, mascara):
    """Que fração da fruta está manchada? (0.0 = nada, 1.0 = tudo)"""
    pixels_fruta = max(cv2.countNonZero(mascara), 1)
    return cv2.countNonZero(mascara_manchas(hsv, mascara)) / pixels_fruta
