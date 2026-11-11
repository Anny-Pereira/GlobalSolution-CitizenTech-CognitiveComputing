"""
BAIXAR FOTOS PRONTAS (banana, maçã e uva: saudável e podre)
===========================================================
Baixa um dataset público do Kaggle com fotos REAIS de frutas saudáveis e podres
e copia N fotos de cada classe (padrão: 30) para a pasta dataset/ no formato
que o projeto usa:

    dataset/banana/saudavel/   dataset/banana/podre/
    dataset/maca/saudavel/     dataset/maca/podre/
    dataset/uva/saudavel/      dataset/uva/podre/

Dataset usado: "Fruit and Vegetable Disease (Healthy vs Rotten)"
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
(Citem esse link no relatório, na seção Dataset e nas Referências.)

Como rodar (a partir da pasta do projeto):
    pip install kagglehub
    python src/baixar_fotos.py                 # 30 fotos por classe
    python src/baixar_fotos.py --quantidade 100

Se o download automático não funcionar, baixem o .zip manualmente pelo link
acima (botão "Download"), descompactem e rodem:
    python src/baixar_fotos.py --origem "C:/caminho/da/pasta/descompactada"
"""

import argparse
import random
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DATASET = RAIZ / "dataset"
DATASET_KAGGLE = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Nome da pasta no projeto -> palavra que aparece no nome da pasta do Kaggle
FRUTAS = {"banana": "banana", "maca": "apple", "uva": "grape", "laranja": "orange"}
ESTADOS = {"saudavel": ("healthy", "fresh"), "podre": ("rotten",)}


def baixar_do_kaggle():
    """Baixa o dataset (≈ alguns GB na 1ª vez; depois fica em cache)."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Instale antes: pip install kagglehub")
    print("Baixando do Kaggle (pode demorar na primeira vez)...")
    return Path(kagglehub.dataset_download(DATASET_KAGGLE))


def encontrar_pastas(origem, palavra_fruta, palavras_estado):
    """
    Procura pastas cujo nome tenha a fruta E o estado. Ex.: 'Apple__Rotten'.
    Ignora maiúsculas/minúsculas, então funciona com variações de nome.
    """
    encontradas = []
    for pasta in origem.rglob("*"):
        nome = pasta.name.lower()
        if "pineapple" in nome:  # "pineapple" contém "apple", mas é abacaxi
            continue
        if (pasta.is_dir() and palavra_fruta in nome
                and any(p in nome for p in palavras_estado)):
            encontradas.append(pasta)
    return encontradas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quantidade", type=int, default=30,
                        help="fotos por classe (padrão 30)")
    parser.add_argument("--origem",
                        help="pasta já baixada/descompactada (pula o download)")
    args = parser.parse_args()

    origem = Path(args.origem) if args.origem else baixar_do_kaggle()
    random.seed(42)  # sempre sorteia as mesmas fotos (resultado reproduzível)

    for fruta, palavra_fruta in FRUTAS.items():
        for estado, palavras_estado in ESTADOS.items():
            fotos = []
            for pasta in encontrar_pastas(origem, palavra_fruta, palavras_estado):
                fotos += [f for f in pasta.rglob("*")
                          if f.suffix.lower() in EXTENSOES]
            if not fotos:
                print(f"[aviso] não achei fotos de {fruta}/{estado} em {origem}")
                continue

            escolhidas = random.sample(fotos, min(args.quantidade, len(fotos)))
            destino = PASTA_DATASET / fruta / estado
            destino.mkdir(parents=True, exist_ok=True)
            for i, foto in enumerate(escolhidas, start=1):
                shutil.copy(foto, destino / f"{fruta}_{estado}_{i:03d}{foto.suffix.lower()}")
            print(f"{fruta}/{estado}: {len(escolhidas)} fotos copiadas "
                  f"(de {len(fotos)} disponíveis)")

    print(f"\nPronto! Fotos em {PASTA_DATASET}. Agora rode: python src/calibrar.py")


if __name__ == "__main__":
    main()
