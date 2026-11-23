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
      Synthetic Fixture Registry
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
[CI CPU/preflight](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45252056405) e na [CI CUDA-source](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45250739007); os novos
testes de versionamento e build do Docker estão incorporados à CI.
**Critério de aceite:** CI CPU e CUDA-source passam;
`sandbox_preflight.py` e `docker compose config` passam sem GPU;
ambientes e scripts locais estão documentados. **O teste da RTX 5070 Ti
continua pendente como validação de hardware do futuro executor, não como
prova falsa de execução atual.** Cheque a execução em
[docs/phase1-design.md](docs/phase1-design.md).

## Fase 2 — Infraestrutura de fixtures, testes e oráculos [PRÓXIMA]

**Objetivo:** montar um harness de testes capaz de aceitar uma futura
implementação de algoritmo e distinguir erro, desempenho e efeito espacial
sem precisar escrever o algoritmo. Detalhamento:
[docs/phase2-test-harness.md](docs/phase2-test-harness.md).

### 2.A — Catálogo e contratos de fixtures
- [ ] Definir manifest JSON versionado: `case_id`, `seed`, `model_id`,
  `input_shape`, `dtype`, `layout`, `device`, `tags`, `expected`,
  `tolerances`, `source_hash`.
- [ ] Fixtures determinísticas para GEMM, residual, softmax, LayerNorm, GELU,
  QKV/atenção, patch projection e conversão de layout.
- [ ] Casos numéricos: zeros, constantes, extremos de logits, NaN/Inf,
  over/underflow, dimensões inválidas, dtype incompatível, padding e máscaras.
- [ ] Casos de escala de atenção: tokens 64/128/224/1.370, múltiplas heads
  e dimensão 64 por head, sem supor divisibilidade de tiles.

### 2.B — Oráculo e comparação
- [ ] Criar interface `ReferenceOracle` (Rust CPU / Hugging Face opcional).
- [ ] Criar `CandidateExecutor` injetável (mock inicialmente), com
  identificador de versão e indicação do backend realmente executado.
- [ ] Comparar shape, dtype, finitude, `max_abs`, `relative_L2`, e
  opcionalmente erro por token e por camada; tolerâncias pré-definidas.
- [ ] Testes que falham intencionalmente ao alterar um valor, trocar dois
  patches, ignorar CLS, alterar normalização ou provocar fallback.

### 2.C — Geometria e espacialidade (dados sintéticos)
- [ ] Gerar imagens/tiles e máscaras **sintéticas** com coordenadas
  conhecidas; não acessar dados de pacientes para desenvolver o harness.
- [ ] Conferir transformação imagem→crop→patch→mapa CHW e inversão;
  registro por nível/escala com `micrometers_per_pixel` opcional.
- [ ] Testar que o harness detecta perda de regiões marcadas após crop,
  resize ou seleção de tokens, sem implementar o algoritmo de seleção.

### 2.D — CI e relatórios mínimos
- [ ] Rodar testes puros no GitHub Actions sem checkpoint nem GPU.
- [ ] Manter a paridade V0 pesada em workflow opcional/reprodutível.
- [ ] Gerar relatório JSON de cada caso com seed, forma, erro e status;
  bloquear regressões reais de correção sem depender de timing instável.

**Gate da Fase 2:** inserir um algoritmo *mock* que (1) passa quando
equivalente e (2) falha nos casos propositalmente corrompidos, inclusive
perda de espacialidade. Nenhum algoritmo de pesquisa é implementado.

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

## Fase 5 — Validação de metadados OSCC e geometria [PLANEJADA]

- [ ] Schema **de metadados**, não ingestão de pacientes: paciente/slide,
  aquisição, magnificação, `µm/px`, origem, coordenadas, rótulo disponível,
  licença, hash e origem de anotação.
- [ ] Fixtures artificiais para conflito de IDs, slides duplicados,
  partições com vazamento e máscaras desalinhadas.
- [ ] Testes de grupos patient/slide-safe e registro espacial de patches
  com coordenadas sintéticas.
- [ ] Testar política de privacidade: dados, pesos e tensores derivados
  não podem ser commitados como artefatos de CI.
- [ ] Especificar apenas métricas/campos esperados para futuros estudos;
  não treinar head, não usar anotações reais nesta branch.

**Gate:** validadores detectam entradas sintéticas inválidas sem dados
clínicos reais.

## Fase 6 — Registro e reprodutibilidade de experimentos [PLANEJADA]

- [ ] Manifest imutável: Git SHA, versão do kernel, modelo/checkpoint hash,
  hash do dataset/fixture, hardware, driver, CUDA, Rust, config e seed.
- [ ] API/CLI para coletar e comparar resultados entre duas execuções,
  com distinção `CPU-CI`, `GPU-local` e `source-only`.
- [ ] Relatórios versionados JSON/CSV e resumo Markdown, sem dados privados.
- [ ] Regression gates para corretude e relatórios de benchmark; tempo
  instável não pode causar falso negativo no CI.
- [ ] Guardar resultados negativos, OOM e exclusões com causa explícita.

**Gate:** executar todo o pipeline sintético e reproduzir o mesmo relatório.

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

## Fora desta branch

Forward DINOv2 completo em CUDA, novos kernels de otimização, mudança de
atenção, pruning de tokens, treinamento de classificadores, adaptação de
Hibou-B/Phikon-v2 reais, inspeção de imagens de pacientes, inferência
diagnóstica, experimentos clínicos, análises de resultados científicos ou
comprovação de ganho de desempenho. **Esses itens foram retirados do
roadmap de sandbox e serão planejados em branches de implementação.**

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
