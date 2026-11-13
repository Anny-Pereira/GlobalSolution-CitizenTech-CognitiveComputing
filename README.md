# Participantes
Anny Pereira               RM553793
Katherine Fernandes        RM552673

# Classificador de Frutas para Agricultores

Programa de **Visão Computacional** que olha a foto de uma fruta e diz:

1. **qual é a fruta** (banana, maçã ou uva);
2. **se ela está saudável ou podre**;
3. **o que fazer** (vender/armazenar ou descartar).

Se a cor da foto não parecer com nenhuma fruta conhecida, ele mostra
"Fruta não identificada".

> Global Solution 2026 (CitizenTech, eixo \*Tecnologia no campo\*) — Cognitive
> Computing, Computer Vision and IoT Systems (Prof. Arnaldo Viana).
> ODS 2, 9 e 10.

## Como funciona

Usa só **Processamento de Imagem com OpenCV**, como nas aulas:

1. **Separa a fruta do fundo** (HSV, máscara, contorno).
2. **Mede a cor** da fruta (histograma) e **quanto dela está manchado** (`cv2.inRange`).
3. **Aplica regras**:

   * a fruta é aquela com a cor mais parecida (`cv2.compareHist`);
   * se a fruta tiver mais manchas que o limite → **podre**.
4. **Mostra o resultado** na foto ou na webcam.

Os limites das regras são descobertos a partir das fotos do `dataset/`.

## Pastas

```
projeto-frutas/
├── src/            ← código
├── notebooks/      ← passo a passo com imagens
├── dataset/        ← fotos usadas para calibrar as regras
├── analises/       ← coloque aqui as fotos que quer analisar
└── docs/           ← relatório e explicações
```

## Como rodar

**1. Instalar** (só uma vez)

```
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

(No Mac/Linux, a segunda linha é `source .venv/bin/activate`.)

**2. Baixar as fotos** (só uma vez)

```
python src/baixar\_fotos.py
```

**3. Calibrar as regras**

```
python src/calibrar.py
```

Cria o arquivo `regras.json` e os gráficos na pasta `resultados/`.

**4. Usar**

Coloque as fotos na pasta `analises/` e rode:

```
python src/prever.py
```

Cada foto abre numa janela; aperte qualquer tecla para ir para a próxima.
Para usar a webcam: `python src/prever.py --webcam` (aperte **Q** para sair).

## Resultados

Os números e gráficos estão na pasta `resultados/` e a análise está em
`docs/relatorio.pdf`.

