"""
ETAPA 3 - CALIBRAR AS REGRAS COM AS FOTOS
=========================================
Aqui NÃO tem Machine Learning. O programa só usa as fotos do dataset/ para
descobrir os números das regras:

  Regra 1 - QUAL É A FRUTA:
      Calcula a "impressão digital de cor" média de cada fruta
      (média dos histogramas). Uma foto nova é comparada com cada uma, e
      vence a mais parecida.

  Regra 2 - SAUDÁVEL OU PODRE:
      "Se mais de X% da fruta estiver manchada -> podre".
      O programa testa vários valores de X (0%, 1%, 2%... 100%) e fica
      com o que mais acerta nas fotos. Cada fruta tem o seu X, porque uma uva
      roxa é naturalmente mais escura que uma banana.

  Regra 3 - FRUTA NÃO IDENTIFICADA:
      Se a cor da foto for diferente demais de todas as frutas conhecidas,
      o programa não arrisca um nome. O limite também sai das fotos.

Para saber se as regras funcionam, separamos as fotos em dois grupos:
  - 80% para CALIBRAR (descobrir os números)
  - 20% para TESTAR (fotos que as regras nunca viram)

Como rodar (a partir da pasta do projeto):
    python src/treinar.py
"""

import json
import random
from pathlib import Path

import numpy as np

from caracteristicas import converter_hsv, histograma_cor, percentual_manchas, semelhanca
from preprocessamento import preprocessar

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DATASET = RAIZ / "dataset"
ARQUIVO_REGRAS = RAIZ / "regras.json"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 1. Ler as fotos e medir cada uma
# ---------------------------------------------------------------------------
def ler_fotos():
    """
    Percorre dataset/<fruta>/<estado>/ e mede cada foto.
    O nome das pastas diz a resposta certa (ex.: banana / podre).
    """
    fotos = []
    for caminho in sorted(PASTA_DATASET.glob("*/*/*")):
        if caminho.suffix.lower() not in EXTENSOES:
            continue
        try:
            imagem, mascara, _ = preprocessar(caminho)
        except FileNotFoundError:
            print(f"  [aviso] não consegui abrir: {caminho.name}")
            continue
        hsv = converter_hsv(imagem)
        fotos.append({
            "caminho": caminho,
            "fruta": caminho.parent.parent.name,
            "estado": caminho.parent.name,
            "histograma": histograma_cor(hsv, mascara),
            "manchas": percentual_manchas(hsv, mascara),
        })
    if not fotos:
        raise SystemExit("Nenhuma foto em dataset/. Rode antes: "
                         "python src/baixar_fotos.py")
    return fotos


def separar_calibrar_testar(fotos):
    """Sorteia 80% para calibrar e 20% para testar, de cada grupo."""
    random.seed(42)  # sorteio sempre igual -> resultado repetível
    calibrar, testar = [], []
    grupos = {}
    for foto in fotos:
        grupos.setdefault((foto["fruta"], foto["estado"]), []).append(foto)
    for lista in grupos.values():
        random.shuffle(lista)
        corte = max(1, int(len(lista) * 0.8))
        calibrar += lista[:corte]
        testar += lista[corte:]
    return calibrar, testar


# ---------------------------------------------------------------------------
# 2. Descobrir os números das regras
# ---------------------------------------------------------------------------
def melhor_limite(fotos):
    """
    Testa limites de 0% a 100% para a regra "manchas > limite -> podre"
    e devolve o que acerta mais fotos.
    """
    limites = np.arange(0, 1.0, 0.01)
    acertos = [sum((f["manchas"] > limite) == (f["estado"] == "podre")
                   for f in fotos) for limite in limites]
    # Se vários limites empatarem, pega o do MEIO deles (mais seguro).
    empatados = [l for l, a in zip(limites, acertos) if a == max(acertos)]
    return round(float(np.median(empatados)), 2)


def calibrar_regras(fotos):
    regras = {"cor_media": {}, "limite_manchas": {}}
    notas_proprias = []
    for fruta in sorted({f["fruta"] for f in fotos}):
        da_fruta = [f for f in fotos if f["fruta"] == fruta]
        # Regra 1: impressão digital média = média dos histogramas
        media = np.mean([f["histograma"] for f in da_fruta], axis=0)
        regras["cor_media"][fruta] = media.tolist()
        notas_proprias += [semelhanca(f["histograma"], np.float32(media))
                           for f in da_fruta]
        # Regra 2: melhor limite de manchas para esta fruta
        regras["limite_manchas"][fruta] = melhor_limite(da_fruta)
    # Limite geral (para fruta não identificada)
    regras["limite_manchas"]["geral"] = melhor_limite(fotos)
    # Regra 3 - FRUTA NÃO IDENTIFICADA:
    # vemos o quanto as fotos de cada fruta se parecem com a sua própria cor
    # média. O limite é escolhido para que 95% delas fiquem ACIMA dele.
    # Uma foto abaixo do limite é diferente demais -> "nao identificada".
    regras["limite_semelhanca"] = round(float(np.percentile(notas_proprias, 5)), 2)
    return regras


# ---------------------------------------------------------------------------
# 3. Aplicar as regras (o prever.py usa esta mesma função)
# ---------------------------------------------------------------------------
def aplicar_regras(histograma, manchas, regras):
    """
    Devolve (fruta, semelhança, estado).
    Se nenhuma fruta for parecida o suficiente -> "nao identificada".
    """
    notas = {fruta: semelhanca(histograma, np.float32(media))
             for fruta, media in regras["cor_media"].items()}
    fruta = max(notas, key=notas.get)
    nota = notas[fruta]
    if nota < regras["limite_semelhanca"]:
        fruta = "nao identificada"
    limite = regras["limite_manchas"].get(fruta,
                                          regras["limite_manchas"]["geral"])
    estado = "podre" if manchas > limite else "saudavel"
    return fruta, nota, estado


# ---------------------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------------------
def main():

    print("Lendo e medindo as fotos... (pode levar alguns segundos)")
    fotos = ler_fotos()
    calibrar, testar = separar_calibrar_testar(fotos)
    print(f"{len(fotos)} fotos: {len(calibrar)} para calibrar e "
          f"{len(testar)} para testar.")

    regras = calibrar_regras(calibrar)
    print("\nRegras descobertas:")
    for fruta, limite in regras["limite_manchas"].items():
        print(f"  {fruta}: podre se mais de {limite:.0%} da fruta tiver manchas")
    print(f"  Fruta nao identificada se a cor for menos de "
          f"{regras['limite_semelhanca']:.0%} parecida com as conhecidas")

    # Testa as regras nas fotos que elas nunca viram
    acerto_fruta = acerto_estado = 0
    for foto in testar:
        fruta, _, estado = aplicar_regras(foto["histograma"], foto["manchas"],
                                          regras)
        acerto_fruta += fruta == foto["fruta"]
        acerto_estado += estado == foto["estado"]

    n = len(testar)
    texto = (f"Fotos de teste: {n}\n"
             f"Qual é a fruta: acertou {acerto_fruta} de {n} "
             f"({acerto_fruta / n:.0%})\n"
             f"Saudável ou podre: acertou {acerto_estado} de {n} "
             f"({acerto_estado / n:.0%})\n")
    print("\nTeste com fotos que as regras nunca viram:")
    print(texto)

    # Salva as regras (o prever.py lê este arquivo)
    ARQUIVO_REGRAS.write_text(json.dumps(regras), encoding="utf-8")
    print("Pronto! Agora já dá para usar: python src/prever.py")


if __name__ == "__main__":
    main()
