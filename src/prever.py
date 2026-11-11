"""
ETAPA 4 - USAR O PROGRAMA
=========================
Olha uma foto (ou a webcam) e diz: qual é a fruta, se está saudável ou podre
e o que o agricultor deve fazer.

Como rodar (a partir da pasta do projeto):
    python src/prever.py                     (analisa as fotos da pasta analises/)
    python src/prever.py --imagem foto.jpg   (analisa uma foto só)
    python src/prever.py --webcam            (aperte Q para sair)
"""

import argparse
import json
from pathlib import Path

import cv2

from treinar import aplicar_regras
from caracteristicas import converter_hsv, histograma_cor, percentual_manchas
from preprocessamento import preprocessar

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_REGRAS = RAIZ / "regras.json"
PASTA_ANALISES = RAIZ / "analises"  # coloquem aqui as fotos para analisar
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# O que o agricultor deve fazer em cada estado.
RECOMENDACOES = {
    "saudavel": "Boa para venda e armazenamento.",
    "podre": "Descartar ou compostagem.",
}
# Cor da moldura (B, G, R): verde = saudável, vermelho = podre.
CORES = {"saudavel": (0, 200, 0), "podre": (0, 0, 255)}


def carregar_regras():
    if not ARQUIVO_REGRAS.exists():
        raise SystemExit("Regras não encontradas. Rode antes: "
                         "python src/treinar.py")
    return json.loads(ARQUIVO_REGRAS.read_text(encoding="utf-8"))


def analisar(imagem_ou_caminho, regras):
    """
    Passo a passo:
      1) separa a fruta do fundo
      2) mede a cor e as manchas
      3) aplica as regras -> fruta, estado e recomendação
    """
    imagem, mascara, contorno = preprocessar(imagem_ou_caminho)
    hsv = converter_hsv(imagem)
    manchas = percentual_manchas(hsv, mascara)
    fruta, nota, estado = aplicar_regras(histograma_cor(hsv, mascara), manchas,
                                         regras)
    textos = [
        "Fruta nao identificada" if fruta == "nao identificada"
        else f"{fruta} (cor {nota:.0%} parecida)",
        f"{estado} (manchas: {manchas:.0%})",
        RECOMENDACOES[estado],
    ]
    return imagem, contorno, estado, textos


def desenhar(imagem, contorno, estado, textos):
    """Desenha o contorno da fruta e escreve o resultado na imagem."""
    saida = cv2.resize(imagem, (400, 400))
    escala = 400 / imagem.shape[0]
    cor = CORES[estado]
    cv2.drawContours(saida, [(contorno * escala).astype(int)], -1, cor, 2)
    for i, texto in enumerate(textos):
        y = 25 + i * 25
        # cv2.putText não desenha acentos, por isso os textos não têm acento.
        cv2.putText(saida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4)  # contorno preto para facilitar a leitura
        cv2.putText(saida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    cor, 2)
    return saida


def mostrar(janela, imagem):
    """Abre uma janela com a imagem (sem tela, como no Colab, não faz nada)."""
    try:
        cv2.imshow(janela, imagem)
        return True
    except cv2.error:
        return False


def fechar_janelas():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def analisar_foto(caminho, regras):
    """Analisa UMA foto: escreve no terminal e mostra numa janela."""
    imagem, contorno, estado, textos = analisar(caminho, regras)
    print(f"\nFoto: {Path(caminho).name}")
    for texto in textos:
        print(f"  {texto}")
    if mostrar("Resultado (aperte qualquer tecla)",
               desenhar(imagem, contorno, estado, textos)):
        cv2.waitKey(0)


def analisar_pasta(regras):
    """Analisa TODAS as fotos da pasta analises/, uma por uma."""
    fotos = sorted(f for f in PASTA_ANALISES.glob("*")
                   if f.suffix.lower() in EXTENSOES)
    if not fotos:
        raise SystemExit("Nenhuma foto na pasta analises/. "
                         "Coloquem lá as fotos que querem analisar.")
    print(f"{len(fotos)} foto(s) na pasta analises/. "
          "Aperte qualquer tecla na janela para ir para a próxima.")
    for foto in fotos:
        analisar_foto(foto, regras)
    fechar_janelas()


def usar_webcam(regras, camera=0):
    captura = cv2.VideoCapture(camera)
    if not captura.isOpened():
        raise SystemExit("Não consegui abrir a webcam.")
    print("Aponte a fruta para a câmera. Aperte Q para sair.")
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        imagem, contorno, estado, textos = analisar(quadro, regras)
        if not mostrar("Classificador de frutas",
                       desenhar(imagem, contorno, estado, textos)):
            print("Sem tela disponível para mostrar a webcam.")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    captura.release()
    fechar_janelas()


def main():
    parser = argparse.ArgumentParser(description="Classificador de frutas")
    parser.add_argument("--imagem", help="caminho de uma foto")
    parser.add_argument("--webcam", action="store_true", help="usar a webcam")
    parser.add_argument("--camera", type=int, default=0,
                        help="número da câmera (padrão 0)")
    args = parser.parse_args()

    regras = carregar_regras()
    if args.webcam:
        usar_webcam(regras, args.camera)
    elif args.imagem:
        analisar_foto(args.imagem, regras)
        fechar_janelas()
    else:
        analisar_pasta(regras)


if __name__ == "__main__":
    main()
