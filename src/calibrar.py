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
    python src/calibrar.py
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np

from medidas import converter_hsv, histograma_cor, percentual_manchas, semelhanca
from preprocessamento import carregar_imagem, preprocessar

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DATASET = RAIZ / "dataset"
PASTA_RESULTADOS = RAIZ / "resultados"
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
# 4. Gráficos para o relatório
# ---------------------------------------------------------------------------
def grafico_manchas(fotos, regras):
    """
    Para cada fruta: quanto de mancha têm as saudáveis (verde) e as podres
    (vermelho), e onde ficou o limite (linha preta). Mostra a regra 2.
    """
    frutas = sorted(regras["cor_media"])
    fig, axs = plt.subplots(1, len(frutas), figsize=(5 * len(frutas), 4),
                            squeeze=False)
    for ax, fruta in zip(axs[0], frutas):
        for estado, cor in (("saudavel", "green"), ("podre", "red")):
            valores = [f["manchas"] * 100 for f in fotos
                       if f["fruta"] == fruta and f["estado"] == estado]
            ax.hist(valores, bins=20, range=(0, 100), alpha=0.6, color=cor,
                    label=estado)
        limite = regras["limite_manchas"][fruta] * 100
        ax.axvline(limite, color="black", linestyle="--",
                   label=f"limite = {limite:.0f}%")
        ax.set_title(fruta)
        ax.set_xlabel("% da fruta com manchas")
        ax.set_ylabel("quantidade de fotos")
        ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "manchas_por_fruta.png", dpi=120)
    plt.close(fig)


def grafico_tabela_acertos(certos, previstos, titulo, arquivo):
    """
    Tabela de acertos e erros: linhas = resposta certa, colunas = o que o
    programa disse. A diagonal são os acertos.
    """
    linhas = sorted(set(certos))
    colunas = sorted(set(certos) | set(previstos))
    tabela = np.zeros((len(linhas), len(colunas)), dtype=int)
    for certo, previsto in zip(certos, previstos):
        tabela[linhas.index(certo), colunas.index(previsto)] += 1
    fig, ax = plt.subplots(figsize=(1.6 * len(colunas) + 2,
                                    1.4 * len(linhas) + 1.5))
    ax.imshow(tabela, cmap="Greens")
    for i in range(len(linhas)):
        for j in range(len(colunas)):
            ax.text(j, i, tabela[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks(range(len(colunas)), colunas, rotation=30)
    ax.set_yticks(range(len(linhas)), linhas)
    ax.set_xlabel("o programa disse")
    ax.set_ylabel("resposta certa")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / arquivo, dpi=120)
    plt.close(fig)


def grafico_erros(erros):
    """Mostra as fotos de teste em que o programa errou."""
    if not erros:
        return
    erros = erros[:8]
    colunas = min(4, len(erros))
    linhas = int(np.ceil(len(erros) / colunas))
    fig, axs = plt.subplots(linhas, colunas, figsize=(3.2 * colunas, 3.4 * linhas),
                            squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (foto, fruta, estado) in zip(axs.flat, erros):
        ax.imshow(carregar_imagem(foto["caminho"])[:, :, ::-1])  # BGR -> RGB
        ax.set_title(f"certo: {foto['fruta']}/{foto['estado']}\n"
                     f"disse: {fruta}/{estado}", fontsize=9, color="red")
    fig.suptitle("Fotos de teste em que o programa errou")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "fotos_com_erro.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------
def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)

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
    certos_fruta, ditos_fruta, certos_estado, ditos_estado, erros = [], [], [], [], []
    for foto in testar:
        fruta, _, estado = aplicar_regras(foto["histograma"], foto["manchas"],
                                          regras)
        acerto_fruta += fruta == foto["fruta"]
        acerto_estado += estado == foto["estado"]
        certos_fruta.append(foto["fruta"]); ditos_fruta.append(fruta)
        certos_estado.append(foto["estado"]); ditos_estado.append(estado)
        if fruta != foto["fruta"] or estado != foto["estado"]:
            erros.append((foto, fruta, estado))

    n = len(testar)
    texto = (f"Fotos de teste: {n}\n"
             f"Qual é a fruta: acertou {acerto_fruta} de {n} "
             f"({acerto_fruta / n:.0%})\n"
             f"Saudável ou podre: acertou {acerto_estado} de {n} "
             f"({acerto_estado / n:.0%})\n")
    print("\nTeste com fotos que as regras nunca viram:")
    print(texto)

    # Salva as regras (o prever.py lê este arquivo) e os resultados
    ARQUIVO_REGRAS.write_text(json.dumps(regras), encoding="utf-8")
    (PASTA_RESULTADOS / "resultado.txt").write_text(texto, encoding="utf-8")
    grafico_manchas(calibrar, regras)
    grafico_tabela_acertos(certos_fruta, ditos_fruta, "Qual é a fruta",
                           "acertos_fruta.png")
    grafico_tabela_acertos(certos_estado, ditos_estado, "Saudável ou podre",
                           "acertos_estado.png")
    grafico_erros(erros)

    print("Pronto! Agora já dá para usar: python src/prever.py")
    print("Gráficos para o relatório estão na pasta resultados/")


if __name__ == "__main__":
    main()
