# ViT Research Sandbox — Environment, Contracts, Tests & Benchmarks

**Branch:** `feat/testing-background`  
**Scope:** preparar uma sandbox reprodutível para investigar, no futuro,
otimizações matemáticas em backbones visuais (inicialmente DINOv2 Base) e
seu impacto sobre padrões histopatológicos de OSCC.  
**Hardware-alvo:** RTX 5070 Ti (sm_120) + Ryzen 7; características do host
devem ser registradas por scripts, não presumidas.  
**Toolchain-alvo:** Rust 2024, CUDA 13.3, NVIDIA CUDA Rust Tile
(`cutile = "=0.5.0"`), `ndarray` CPU como oráculo.

**Regra de escopo:** nesta branch implementamos SOMENTE configuração,
ambiente, scaffolding, contratos, fixtures, oráculos, testes, benchmarks,
CI, relatórios e mecanismos de comparação. **Não implementamos aqui novos
algoritmos de otimização, treinamento supervisionado, classificação de câncer,
segmentação clínica, nem o forward CUDA completo do modelo.** Esses são
trabalhos de futuras branches de experimentos, após a sandbox estar pronta.

**Semântica:** `[x]` significa tarefa concluída com código/configuração e
evidência de CI, quando aplicável. Compilação não demonstra execução na GPU;
uma comparação numérica não demonstra qualidade clínica.

## Arquitetura da sandbox

```text
Versioned Experiment Manifest / Seed / Model + Data Hash
                │
      Existing-image source registry + numeric operator fixtures
                │
    Reference Model / CPU Oracle  ─────────────┐
                │                              │
     Algorithm Candidate Interface             │
      (empty / mock / future work)             │
                │                              │
       CPU/GPU Backend Contracts               │
                │                              │
  Correctness Gate ← Compare Identical Inputs ┘
                │
    Runtime Profile (GPU / CPU / transfer)
                │
  Benchmark Repetitions / A-B Comparison
                │
  Versioned JSON/CSV Reports + Regression Gate
```

No benchmark pode atribuir melhoras a um algoritmo se o modelo, pesos, seed,
dataset, pré-processamento e hardware não forem controlados.

## V0 — Referência FP32 e tokens espaciais [CONCLUÍDA]

- [x] Base independente Rust/ndarray com checkpoint original DINOv2 Base.
- [x] `DenseFeatures` preserva CLS, 1.369 patches e mapa `[768,37,37]`.
- [x] Pré-processamento RGB e coordenadas de crop/patch testados.
- [x] Fixtures de contrato, CLI, Rustfmt e Clippy aprovados.
- [x] Oráculo Hugging Face com checkpoint e versões fixadas.
- [x] Paridade de CLS, 1.369 patches e mapa CHW demonstrada na
  [CI real](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357).

**Evidência:** [docs/v0-validation.md](docs/v0-validation.md).
É uma referência matematicamente comparável, não um sistema de diagnóstico.

## Fase 1 — Provisionamento Rust / CUDA e smoke tests da sandbox [CONCLUÍDA]

**Objetivo:** disponibilizar dois ambientes isolados e reproduzíveis, com
comandos de build, checagem do projeto e testes de infraestrutura.
**Não exige implementar outro forward neural nem executar uma melhoria
algorítmica.** Ver [PHASE1.md](PHASE1.md) e
[docs/phase1-design.md](docs/phase1-design.md).

- [x] Separar imagens `Dockerfile.cpu` (Rust CPU) e `Dockerfile`
  (CUDA 13.3 + Rust), mantendo Compose com perfis independentes.
- [x] Fixar o `cutile` GPU, os `Cargo.lock` e a toolchain Rust **1.100.0**
  em configuração central, Docker e workflows.
- [x] Oferecer contratos de shape, dtype, layout e device para os operadores
  a serem comparados, com rejeição explícita de fallback GPU→CPU.
- [x] Adicionar kernels **smoke/fixture** (residual, softmax, GEMM) para
  testar toolchain; não são alterações de algoritmo de pesquisa.
- [x] Instalar CI CUDA-source sem GPU e CI CPU com testes, Rustfmt,
  Clippy, CLI, `--locked`, verificação de referências obsoletas e
  build automático da imagem CPU.
- [x] Adicionar script de preflight offline, testes automáticos de configuração,
  verificação de sintaxe Bash e validação da configuração Docker Compose.
- [x] Disponibilizar script local para verificar e registrar GPU, driver,
  toolkit, Rust, commit e saída dos smoke tests **sem fallback**.
- [x] Documentar env `DATA_DIR`, `MODEL_DIR`, SELinux Fedora, comandos de
  execução e a distinção entre `source-ready` e `gpu-verified`.

**Encerramento da Fase 1:** validação confirmada na
[CI CPU/preflight](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400561) e na [CI CUDA-source](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400586); os novos
testes de versionamento e build do Docker estão incorporados à CI.
**Critério de aceite:** CI CPU e CUDA-source passam;
`sandbox_preflight.py` e `docker compose config` passam sem GPU;
ambientes e scripts locais estão documentados. **O teste da RTX 5070 Ti
continua pendente como validação de hardware do futuro executor, não como
prova falsa de execução atual.** Cheque a execução em
[docs/phase1-design.md](docs/phase1-design.md).

## Fase 2 — Harness sobre o acervo ORCA 512×512 [PRÓXIMA]

**Objetivo:** preparar testes e comparadores para receber futuramente
um algoritmo, usando exclusivamente as imagens e máscaras existentes,
sem implementar melhorias, novos modelos ou treinamento. O corpus real
de entrada já está inventariado em
[docs/orca-dataset-audit.md](docs/orca-dataset-audit.md), com
[CI de auditoria dos 400 PNGs](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098).
Especificação detalhada: [Fase 2](docs/phase2-test-harness.md).

### 2.0 — Inventário de dados [AUDITADO]

- [x] Localizar os **200 pares** RGB/`_mask.png` em
  `ORCA_512x512/{training,testing}`, 100 pares em cada split.
- [x] Confirmar em CI: **400 PNGs válidos**, RGB8 e grayscale8
  de 512×512, com hashes, pareamento completo e nenhum conteúdo duplicado.
- [x] Identificar **101 IDs de caso TCGA** pelos filenames,
  51 training/50 testing, **sem overlap de IDs entre splits**.
- [x] Verificar diretamente todos os pixels: **200 máscaras binárias
  com intensidades 0 e 255**, diferentemente do ORCA original
  descrito na literatura com três classes.
- [x] Quantificar o risco do crop do pré-processamento V0:
  **8,78% dos pixels 255 fora do crop central equivalente**,
  59/200 máscaras com perda estimada superior a 10%.
- [ ] Documentar a procedência da variante binária, significado
  validado de 0/255, atribuição, condições de uso e redistribuição.
- [ ] Congelar um manifest local de todos os pares com hash SHA-256,
  caso TCGA, core, split e versão de pré-processamento.

### 2.A — Manifest e catálogo de testes

- [ ] Schema JSON versionado para `case_id`, `core_id`, `split`,
  `image_path`, `mask_path`, `image_sha256`, `mask_sha256`,
  `mask_values=[0,255]`, shape 512×512, seed, dtype/layout,
  modelo, backend, tolerâncias e origem da anotação.
- [ ] Parser read-only de imagens e máscaras, contrato de
  pareamento e verificação de imagem/máscara trocadas ou ausentes.
- [ ] Fixtures **numéricas** para GEMM, residual, softmax,
  LayerNorm, GELU, QKV e atenção; testes NaN/Inf, shapes e padding.
- [ ] Preservar os splits já enviados e bloquear mistura de cores
  do mesmo caso TCGA entre treino e teste.

### 2.B — Oráculo e comparador numérico

- [ ] `ReferenceOracle` CPU (V0 e operações isoladas) e
  `CandidateExecutor` mock, sem implementar um novo algoritmo.
- [ ] Comparar shape, dtype, finitude, `max_abs`,
  `relative_L2`, layout e ordens de CLS/patches; registrar seed e versão.
- [ ] Testes negativos: valor perturbado, token permutado, NaN,
  divergência além da tolerância e fallback GPU→CPU falso.

### 2.C — Máscaras ORCA e registro espacial

- [ ] Aplicar o perfil V0 **512→592→center crop 518**
  à imagem e à máscara pareada, com CatmullRom para RGB
  e **nearest-neighbor na máscara**, preservando 0/255.
- [ ] Projetar a máscara transformada à grade **37×37**
  de patches 14×14, computando fração de pixels 255 por patch
  sem tratar CLS como patch e **sem inventar rótulos clínicos**.
- [ ] Medir, reportar e testar **cobertura de anotação preservada
  e perdida** em crop/resize; usar 8,78% como baseline
  geométrico aproximado, não limite de aceite clínico.
- [ ] Detectar trocas de pares, máscara deslocada, nova classe
  de pixel por interpolação incorreta, grade transposta e
  contaminação entre IDs de caso.
- [ ] Limitar coordenadas à imagem TMA 512×512, marcando
  WSI x/y, nível e µm/px como desconhecidos até confirmação.

### 2.D — CI e resultados do harness

- [x] Workflow independente de auditoria dos 400 PNGs
  com relatório agregado e **sem publicar imagens**
  ([evidência](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098)).
- [ ] Suíte CPU de contratos e corrupção controlada de
  dados/máscaras existentes, sem dependência de GPU nem pesos.
- [ ] Reporter JSON por caso com hashes, classes 0/255,
  matriz de transformação, cobertura e motivo de falha.
- [ ] Manter comparações pesadas do checkpoint V0 em CI separada.

**Gate da Fase 2:** imagem e máscara pareadas do ORCA passam;
máscara errada, geometria deslocada, classe inválida, split
incompatível e valor numérico corrompido **falham**.
O resultado é uma infraestrutura de testes, não um algoritmo de
segmentação nem validação diagnóstica.


## Fase 3 — Adaptadores e injeção experimental [PLANEJADA]

- [ ] Definir trait/config `ModelAdapter` com capacidades explícitas:
  patch size, CLS, registers, token order, grade, dtype e checkpoint ID.
- [ ] Criar registro e adaptadores **mock** para pelo menos duas
  configurações de ViT, sem baixar/implementar novos backbones.
- [ ] Interface de troca por operação (`baseline`/`candidate`) com
  política `unsupported → error`, identificação de código e ablação.
- [ ] Teste que confirma mesma fixture/mesmo operador em dois adapters
  compatíveis; falha clara em incompatibilidades.
- [ ] Versionar perfis de pré-processamento e manter V0 fixado como oráculo.

**Gate:** contratos e testes de conformance passam com adaptadores fake;
integrar um modelo real diferente será trabalho experimental futuro.

## Fase 4 — Benchmark e perfis CPU/GPU [PLANEJADA]

- [ ] Harness com warmup, execução sincronizada na GPU, repetições,
  mediana/p50/p95, memória pico, eventos GPU e medição separada de H2D/D2H.
- [ ] Configurar perfis de workload (batch, dtype, tokens, tamanho de tiles,
  CPU workers 1/2/4/6/8), com controle de seed e recursos.
- [ ] A/B pareado, execução intercalada para controlar drift térmico e
  relatório de variância/intervalo; sem alegar speedup com single-run.
- [ ] Perfis de benchmark sem GPU (mock/CPU) no CI e perfil GPU opcional
  executado apenas na máquina apropriada.
- [ ] Testar JSON de timing com dados artificiais e negativas (OOM, erro,
  ausência de driver, warmup insuficiente).

**Gate:** com duas funções sintéticas conhecidas, o harness detecta
ganho/queda e rejeita dados de performance incompletos.

## Fase 5 — Schemas de proveniência e integridade do acervo [PLANEJADA]

- [ ] Completar a política de casos TCGA: dois cores de uma mesma
  origem ficam no mesmo split; manter training/testing originais
  e detectores de duplicatas ou derivados entre grupos.
- [ ] Validar manifest da versão binária ORCA e origem do
  rótulo 255, sem presumir equivalência com as três classes
  descritas nos estudos do dataset original.
- [ ] Definir metadata opcional WSI, nível, magnificação e
  micrômetros por pixel: ausência = desconhecido, não inferido.
- [ ] Verificar permissão de uso/redistribuição e atribuição
  dos arquivos **já commitados no repositório público**;
  registrar decisão sem mudar o dataset automaticamente.
- [ ] Testar proveniência e transformações a partir das imagens
  e máscaras existentes, sem gerar outras imagens/anotações.
- [ ] Especificar apenas formatos e esquemas de métricas para
  experimentos futuros, sem treinar modelos nem medir eficácia clínica.

**Gate:** provenance documentada, splits preservados, artefatos
derivados identificáveis e nenhuma inferência física sem dados.


## Fase 6 — Registro e reprodutibilidade de experimentos [PLANEJADA]

- [ ] Manifest imutável: Git SHA, versão do kernel, modelo/checkpoint hash,
  hash do dataset/fixture, hardware, driver, CUDA, Rust, config e seed.
- [ ] API/CLI para coletar e comparar resultados entre duas execuções,
  com distinção `CPU-CI`, `GPU-local` e `source-only`.
- [ ] Relatórios versionados JSON/CSV e resumo Markdown, sem dados privados.
- [ ] Regression gates para corretude e relatórios de benchmark; tempo
  instável não pode causar falso negativo no CI.
- [ ] Guardar resultados negativos, OOM e exclusões com causa explícita.

**Gate:** repetir relatórios sobre os mesmos operadores numéricos e
exemplos derivados do acervo, com hashes e transformações idênticos.

## Fase 7 — Prontidão da sandbox [PLANEJADA]

- [ ] `make preflight`, `make test`, `make smoke-gpu`,
  `make bench` com documentação de exigências.
- [ ] Runbook Fedora/RTX 5070 Ti com diagnóstico de driver/container toolkit
  e scripts para coletar evidência de execução GPU local.
- [ ] Pipeline end-to-end com plugin algorítmico de exemplo que não modifica
  resultados e outro deliberadamente incorreto.
- [ ] CI CPU completa e CI de compilação GPU, documentação de mocks e falhas.
- [ ] Checklist de isolamento de branches, dados e artefatos: cada nova
  hipótese experimental começa com configuração, baseline, testes e métricas.

**Gate:** um novo experimento consegue conectar um algoritmo futuro ao
harness e obter erros, métricas e benchmark sem reconfigurar o ambiente.

## Evidências e estados

- `source-ready`: estrutura, dependências e código CUDA compilaram.
- `sandbox-config-ready`: mais preflight offline, testes de contrato,
  scripts e Compose validados; pode encerrar Fase 1.
- `gpu-verified`: smoke executado na RTX 5070 Ti com logs reais;
  **não presumir que ocorreu**.
- `experiment-ready`: somente depois da Fase 7 e dos testes de mock.

A CI do projeto publica apenas evidências de testes e infraestrutura. O
resultado de V0 não autoriza afirmações sobre detecção de câncer ou
execução CUDA do forward.
