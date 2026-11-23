# ORCA 512×512 — inventário técnico do acervo

**Auditoria:** 2026-11-23  
**Origem inspecionada:** `ORCA_512x512/` da branch `feat/testing-background`  
**Resultado:** [auditoria aprovada dos 400 PNGs](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098).  
**Auditor:** [`scripts/audit_orca_dataset.py`](../scripts/audit_orca_dataset.py).

## Inventário e integridade

| Split | Imagens RGB8 512×512 | Máscaras grayscale8 512×512 | IDs TCGA distintos |
| --- | ---: | ---: | ---: |
| Training | 100 | 100 | 51 |
| Testing | 100 | 100 | 50 |
| **Total** | **200** | **200** | **101** |

Pastas efetivas:
- `training/tumor/tma/` ↔ `training/lesion_annotations/`;
- `testing/tumor/tma/` ↔ `testing/lesion_annotations/`.

A auditoria leu os arquivos existentes e verificou assinatura,
CRC dos chunks PNG, descompressão IDAT, dimensão, tipo de cor,
pareamento por nomes e repetição por hash de conteúdo. Resultado:
**200 pares válidos, zero imagens sem máscara, zero máscaras órfãs,
zero arquivos repetidos byte a byte, zero IDs TCGA comuns aos splits**.
As 100 imagens de teste correspondem a 50 IDs com dois recortes
cada. Das 100 imagens de treino, 49 IDs têm dois recortes e dois IDs
têm somente um.

Os IDs de caso foram **inferidos dos filenames**; ainda não foram
confirmados contra um manifest de pacientes, WSIs ou aquisições.
Esse dado é suficiente para testar vazamento por prefixo TCGA,
mas não para garantir proveniência clínica completa.

## Classes das 200 máscaras

Cada arquivo `*_mask.png` tem **somente os valores 0 e 255**
e contém pelo menos um pixel de cada valor.

| Intensidade | Pixels totais | Percentual |
| --- | ---: | ---: |
| 0 | 36.262.819 | 69,17% |
| 255 | 16.165.981 | 30,83% |

Pixels 255: **33,37%** no treino e **28,30%** no teste.
São proporções de pixels, não a distribuição de pacientes ou
métricas de eficácia.

A [publicação do ORCA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406800/)
descreve **três classes** (carcinoma branco, tecido não carcinomatoso
cinza, não tecido preto). Esta cópia concreta possui apenas
**duas intensidades**. Portanto o harness deve tratar `0/255` como
**máscara binária**; não inventar a classe cinza, nem equiparar
todo pixel 0 a tecido normal. A origem e o significado exato
da variante binária precisam de validação documental.

## Geometria: 512 original → 592 resize → 518 crop

O pré-processamento V0 usa `RESIZE_SHORTEST=592` e
`INPUT_SIZE=518`. Para entrada 512×512, o recorte central equivale
aproximadamente à caixa original `x,y ∈ [32,480)`,
que cobre 76,56% da área original.

Auditoria sobre as anotações existentes de valor **255**:

| Medida aproximada | Resultado |
| --- | ---: |
| Pixels 255 na região central preservada | 14.746.907 — **91,22%** |
| Pixels 255 fora da região central | 1.419.074 — **8,78%** |
| Máscaras com algum pixel 255 na borda excluída | **199/200** |
| Máscaras com mais de 10% de pixels 255 excluídos | **59/200** |

É uma **estimativa em coordenadas de origem**, não uma medida
após interpolação, nem resultado de inferência ou segmentação.
O harness deverá aplicar o mesmo resize/crop à imagem e à
máscara pareada; usar CatmullRom na imagem conforme V0 e
**nearest-neighbor na máscara** para preservar 0/255. Ao
converter em patches 14×14 da grade 37×37, medir cobertura
da anotação por patch e perda nas bordas, preservando o
mapeamento inverso. Não alterar o algoritmo V0 nesta branch
para esconder a perda; relatá-la explicitamente.

## Proveniência e riscos ainda abertos

- Falta manifest por par com `case_id`/`core_index`,
  `source_image_sha256`, `annotation_sha256`,
  `split`, semântica da variante binária e transformações.
- Não há metadados verificáveis nesta árvore que permitam
  reconstruir origem WSI `x/y`, nível piramidal ou
  `micrometers_per_pixel`. Tratar como desconhecidos até
  haver fonte confiável.
- A estrutura contém TMAs redimensionados, **não WSIs**
  originais completos. Testar geometria relativa a 512×512;
  não afirmar localização física numa lâmina sem provenance.
- **Os 400 PNGs estão rastreados no repositório público**.
  O atual `.gitignore` para `/data/` não os remove do Git.
  Antes de redistribuição adicional, revisar condições do
  [ORCA oficial](https://sites.google.com/unibas.it/orca/home),
  citação e licença; não mover, apagar ou republicar estes
  arquivos sem uma decisão explícita.
- O auditor não prova por si só que a máscara foi alinhada
  clinicamente ou se a intensidade 255 representa uma
  anotação de tumor em cada amostra. Isso requer confirmar
  a origem/transformação da versão binária.

## Consequências para a sandbox

**Fase 2:** já existe corpus de entrada validado.
Faltam manifest, transformador de máscara pareada, parser
de classes, alinhamento imagem↔37×37 e comparadores negativos.

**Fase 5:** conferir split por caso/slide, proveniência física
quando existente e política de uso, sem gerar máscaras.

**Fases 3/4/6:** reutilizar hashes, casos e métricas para
oráculo, comparação A/B e report. Não criar novos modelos
nem alegar diagnóstico clínico nesta branch.

## Reexecutar (somente leitura dos PNGs)

```bash
python3 scripts/audit_orca_dataset.py \
  --dataset ORCA_512x512 \
  --output results/orca-audit.json \
  --expected-pairs 200
```

O script grava apenas métricas agregadas em `results/`,
já ignorado pelo Git. O
[workflow de auditoria](../.github/workflows/orca-dataset-audit.yml)
publica somente o relatório JSON, não as imagens.
