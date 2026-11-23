# Fase 2 — Harness de testes baseado no ORCA efetivamente enviado

**Estado:** inventário de entrada concluído; implementação
de comparadores e testes espaciais pendente.

[Relatório de auditoria dos 200 pares](orca-dataset-audit.md) ·
[GitHub Actions: 400 PNGs verificados](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098)

## Base verificada para os testes

| Campo | Evidência |
| --- | --- |
| Fonte | `ORCA_512x512/` no Git, **400 PNGs** |
| Imagens | **200**, RGB 8-bit 512×512 |
| Máscaras | **200**, grayscale 8-bit 512×512, pareadas 1:1 |
| Classes efetivas | **0 e 255**, somente |
| Splits | Training 100, testing 100 |
| Identificadores | 101 casos TCGA de filenames (51/50), sem overlap |
| Testes existentes | Auditor de CRC/PNG, pairing, hashes, classes e splits |
| Ainda não verificado | Semântica da variante binária, registro após preprocess, escala WSI |

O [ORCA original](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406800/)
é descrito com três classes; a versão do repositório é
**binária**. O valor 0 não distingue tecido normal de
fundo. O harness não pode atribuir classe cinza nem gerar
máscaras. A semântica exata de 255 permanece sujeita à
checagem de origem, embora os arquivos sejam chamados
`lesion_annotations`.

## Fluxo de dados e configuração

- Usar somente `ORCA_512x512/training/tumor/tma` e
  `ORCA_512x512/testing/tumor/tma` e seus pares
  `lesion_annotations/*_mask.png`.
- `DATA_DIR=./ORCA_512x512` pode montar o dataset
  read-only em `/data` no serviço Docker; os arquivos
  já existem e **estão rastreados no Git público**.
- Não criar imagens artificiais ou novas máscaras.
  Vetores e matrizes numéricas de testes isolados de
  operadores podem continuar artificiais.
- Patches, crops e versões de imagem de teste somente
  por derivação rastreável de uma imagem original.
- O auditor deve produzir apenas metadados agregados
  para CI; relatórios por caso ficam em `results/`
  (ignorado pelo Git) sem republicar imagens.

## Entregas ordenadas

| Ordem | Módulo | Teste de aceite |
| --- | --- | --- |
| 2.0 | Inventário/PNG | **Concluído:** 200 pares, formato, hashes, pixels e splits |
| 2.1 | Manifest + loader | 200 pares indexados; rejeitar mask errada, case conflitante, hash inválido |
| 2.2 | Oráculo + executor mock | Mesmo tensor/seed; identidade passa, mutação controlada falha |
| 2.3 | Comparator numérico | Shape/dtype/NaN/`max_abs`/`relative_L2`; ordem CLS+patches |
| 2.4 | Comparator espacial | Resize/crop máscara coerente, grid 37×37, preservação 0/255 |
| 2.5 | JSON report + CI | Detectar todos os testes negativos e preservar logs privados |

### 2.1 Manifest por par

Schema inicialmente exigido: `schema_version`,
`case_id`, `core_index`, `split`,
`image_path`, `mask_path`, `image_sha256`,
`mask_sha256`, `source_width=512`,
`source_height=512`, `mask_value_set=[0,255]`,
`label_schema_version`, `preprocessing_revision`.
Campos WSI `x/y`, nível, magnificação ou
`micrometers_per_pixel` só podem ser preenchidos
com metadados originais confirmados.

A divisão existente é **fixa**; não randomizar por imagem,
pois o mesmo caso pode ter dois recortes. O auditor já
encontrou 101 casos sem overlap entre os splits;
isso precisa ser preservado pelo loader e pelos testes.

### 2.2–2.3 Oráculo, mocks e invariantes numéricas

- `ReferenceOracle` fornece baseline matemático em Rust CPU
  e, opcionalmente, referência Hugging Face V0.
- `CandidateExecutor` fornece versões `identity` e
  `mutated` sem implementar otimizações.
- Conferir mesmo input, seed, backend solicitado/real, shape,
  dtype, layout e ordem de tokens antes dos erros numéricos.
- `max_abs = max(abs(candidate-reference))`;
  `relative_L2 = ||candidate-reference||₂ /
  max(||reference||₂, 1e-12)`; ambas as tolerâncias
  precisam ser satisfeitas.
- Falhar em NaN/Inf, shape/dtype incorreto, troca de
  tokens, divergência deliberada ou fallback CPU oculto.

### 2.4 Contrato espacial específico do ORCA

A imagem RGB V0 passa por **512→592** (CatmullRom),
**center crop 518×518** e divisões em **37×37 patches**
de **14×14**. Repetir exatamente a transformação
geométrica na máscara, mas usando **nearest-neighbor**,
para não inventar classes além de 0/255.

O auditor mediu na imagem original que cerca de
**8,78% dos pixels 255** ficam fora do centro
equivalente; **199/200** máscaras têm alguma anotação
na borda excluída e **59/200** têm mais de 10%
dos pixels 255 fora dessa região. Não alegar
conservação de todas as lesões. Calcular:

1. Coordenadas originais e inversas por pixel/patch;
2. Cobertura de pixels 255 por patch `[37,37]`;
3. Fração de 255 preservada e perdida na transformação;
4. Nenhum token CLS confundido com célula espacial;
5. Alerta/falha quando máscara é trocada, deslocada,
   transposta ou reamostrada com novo valor de classe.

Estas são **verificações de anotação fornecida**,
não previsões de segmentação ou qualidade clínica.

### 2.5 Relatório e CI

Relatório por execução: `schema_version`, `run_id`,
`case_id`, `core_index`, `split`, hashes de origem,
`seed`, `preprocessing_revision`, `requested_device`,
`actual_device`, `shape`, `dtype`, `max_abs`,
`relative_L2`, `mask_classes`, `crop_coverage`,
`passed`, `failure_reason` e `git_sha`.

CI sem GPU: testes numéricos e auditor ORCA.
Não publicar imagens, máscaras, overlays ou novos
derivados nos artefatos. A Fase 2 não usa benchmarks
de tempo como gate; isso será responsabilidade
da Fase 4.

## Testes negativos obrigatórios

- Referência e mock-identidade iguais: **passa**.
- Perturbação numérica além da tolerância: **falha**.
- NaN/Inf, shape ou dtype errado: **falha**.
- Trocar duas células do mapa espacial: **falha**.
- Associar máscara a outra imagem/caso: **falha**.
- Usar interpolação da máscara que produza classe
  diferente de 0/255: **falha**.
- Aplicar crop/offset divergente entre imagem e máscara:
  **falha**.
- Dividir recortes do mesmo caso TCGA entre
  training/testing: **falha**.
- Declarar CUDA mas executar CPU: **falha**.

## Gate

O harness aceita os pares ORCA existentes e rejeita
corrupções deliberadas e mudanças silenciosas de classe,
forma, split e geometria, produzindo relatório JSON.
Não implementar um algoritmo de pesquisa, modelo novo,
segmentador ou treinamento nesta branch.

