# Checklist operacional da sandbox

**Fonte de verdade:** [roadmap.md](roadmap.md). A branch
`feat/testing-background` prepara apenas configurações, testes,
benchmarks e interfaces; não implementa algoritmos de pesquisa.

## V0 — referência numérica [CONCLUÍDA]

- [x] CPU Rust DINOv2 Base, CLS e grid 37×37 validados contra pesos reais.
- [x] Paridade Hugging Face dos 1.370 tokens, valores e layout CHW:
  [evidência](docs/v0-validation.md).

## Fase 1 — ambiente/configuração [CONCLUÍDA]

- [x] Imagens CPU/GPU, perfis Compose e lockfiles separados;
  Rust 1.100.0 fixado e build de imagem CPU integrado à CI.
- [x] Kernel smoke sources de residual, softmax e GEMM para verificar CUDA Rust.
- [x] CUDA 13.3 `cargo check --locked` em CI independente de GPU.
- [x] Contratos CPU de tensor/device + 17 testes Rust.
- [x] Preflight offline, testes de configuração e sintaxe shell.
- [x] Scripts para registrar ambiente e GPU local sem fallback.
- [x] Proteção de dados em `.gitignore`, documentação dos containers.
- [x] Evidências: [CI CPU/preflight](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400561) e [CUDA-source](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400586).

**Verificação externa ainda pendente:** rodar os probes na RTX 5070 Ti
e guardar os resultados. Não é evidência de algoritmo otimizado
nem de inferência do modelo completo. [Fase 1](PHASE1.md).

## Fase 2 — ORCA e harness [AUDITORIA CONCLUÍDA; IMPLEMENTAÇÃO PENDENTE]

- [x] Auditar 400 PNGs ORCA: **200 pares** 512×512 (100 treino /
  100 teste), todos pareados, íntegros e sem duplicatas.
- [x] Verificar classes reais **0/255** nas 200 máscaras,
  101 IDs TCGA sem overlap de treino/teste e
  [evidência](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098).
- [x] Quantificar perda aproximada no crop V0: **8,78%** dos
  pixels 255 excluídos; 59/200 máscaras perdem mais de 10%.
- [ ] Documentar origem da variante binária e autorização
  de redistribuição do ORCA já rastreado no Git.
- [ ] Criar manifest imutável de pares/hashes/TCGA-case/split.
- [ ] Implementar oráculo, executor mock e comparador numérico.
- [ ] Implementar alinhamento imagem/máscara 512→592→518,
  nearest-neighbor na máscara e grade 37×37.
- [ ] Executar testes negativos de máscara trocada, classe
  inválida, crop incorreto e split com vazamento.
- [ ] Relatórios JSON por caso e CI de regressão sem
  publicar imagens/máscaras derivadas.

[Detalhamento Fase 2](docs/phase2-test-harness.md) ·
[Auditoria ORCA](docs/orca-dataset-audit.md).


## Fases 3–7 [PLANEJADAS]

- [ ] Adaptadores e interfaces mock de experimentos.
- [ ] Benchmark A/B e instrumentação CPU/GPU.
- [ ] Validar esquema binário ORCA e proveniência WSI/µm/px
  quando disponível, preservando os dois cores por caso.
- [ ] Proveniência e relatórios reproduzíveis.
- [ ] CLI/runbook e teste end-to-end da sandbox.

**Fora da branch:** forward CUDA completo, otimizações
algorítmicas, novos modelos reais, treinamento, estudo clínico.
