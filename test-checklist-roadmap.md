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

## Fase 2 — harness [PRÓXIMA]

- [ ] Manifest com origem/hash de imagens existentes e fixtures
  **numéricas** versionadas para operações.
- [ ] Interface baseline/candidate/mock e testes de erros deliberados.
- [ ] Comparador `max_abs`, `relative_L2`, shape, dtype, spatial order.
- [ ] Geometria de tiles/crops **derivados das imagens existentes**
  e alinhamento de máscaras somente quando já anotadas.
- [ ] Relatórios JSON e CI de testes sem GPU nem dados privados.

[Detalhamento Fase 2](docs/phase2-test-harness.md).

## Fases 3–7 [PLANEJADAS]

- [ ] Adaptadores e interfaces mock de experimentos.
- [ ] Benchmark A/B e instrumentação CPU/GPU.
- [ ] Validação de schemas e proveniência espacial do acervo existente,
  sem fabricar imagens ou anotações.
- [ ] Proveniência e relatórios reproduzíveis.
- [ ] CLI/runbook e teste end-to-end da sandbox.

**Fora da branch:** forward CUDA completo, otimizações
algorítmicas, novos modelos reais, treinamento, estudo clínico.
