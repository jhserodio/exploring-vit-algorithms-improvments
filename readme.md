# ViT Research Sandbox — Rust / CUDA / Spatial Test Infrastructure

A **sandbox de desenvolvimento e testes** para estudos futuros de
otimizações algorítmicas em modelos visuais (ViT) com aplicação
planejada em histopatologia de OSCC.

**Escopo desta branch `feat/testing-background`:** ambiente,
configuração, interfaces, fixtures sintéticas, validação numérica,
reprodutibilidade, benchmark e CI. **Não** desenvolver nesta branch
novos algoritmos otimizados, treinamento de modelos, inferência
clínica ou avaliação real de câncer.

## Estado atual

- **V0: concluída.** Referência Rust/CPU do DINOv2 Base com CLS
  e grid 37×37 validada contra checkpoint oficial fixado:
  [relatório](docs/v0-validation.md).
- **Fase 1: ambiente configurado e encerrado.** Rust CPU e CUDA Rust em
  containers separados, dependências fixadas, contratos e
  probes de soma residual, softmax e GEMM; CI de compilação CUDA
  sem GPU e CI CPU. Rust 1.100.0 fixado em `rust-toolchain.toml`,
  imagens e workflows. [CI CPU](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400561), [CUDA-source](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45262400586);
  preflight e Compose documentados em [PHASE1.md](PHASE1.md).
- **Fase 2: planejada/detalhada.** Harness de fixtures sintéticas,
  oráculos, comparadores, testes negativos e relatório JSON.
  [Especificação](docs/phase2-test-harness.md).

`source-ready` e `sandbox-config-ready` não significam que
a RTX 5070 Ti já executou os kernels. O estado `gpu-verified`
exige execução no computador local, com os logs correspondentes.
O forward CUDA integral do modelo não faz parte desta branch.

## Ambiente e validação (não requer GPU)

```sh
python3 scripts/sandbox_preflight.py --json
python3 -m unittest discover -s scripts -p 'test_sandbox_preflight.py'
bash -n scripts/*.sh
docker compose --profile gpu config --quiet

cd motor_inferencia
cargo test --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
```

`Dockerfile.cpu` é usado pelo serviço CPU e `Dockerfile` contém
CUDA 13.3 para o serviço GPU:

```sh
docker compose run --build --rm vit-cpu
# Somente com driver NVIDIA, toolkit/container runtime compatível:
docker compose --profile gpu run --build --rm vit-gpu
```

Use diretórios de pesquisa dedicados; volumes `:z` podem alterar
labels SELinux no Fedora, inclusive quando montados como read-only.

## GPU local: instrumentação, não algoritmo de pesquisa

```sh
bash scripts/v1-gpu-acceptance.sh results/v1-gpu
```

O script captura versões de driver, CUDA, Rust, CPU/GPU e Git SHA,
e executa os **smoke tests isolados** para residual, softmax e GEMM.
Falhas de GPU são propagadas, sem fallback para CPU. Também é possível
usar o serviço Docker `vit-gpu` do Compose.

Os probes existem **somente para validar a toolchain e servir como
fixtures**; não representam nova atenção, ganho de desempenho
ou qualquer melhoria de classificação.

## Referência matemática V0

O DINOv2 Base na grade nativa 518px produz:
`CLS=[1,768]`, `patch_tokens=[1,1369,768]`,
`spatial_map=[1,768,37,37]`. A CI de referência
[validou numericamente todos os tokens](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357)
com pesos reais e entrada sintética idêntica ao Hugging Face.
Isso não é um detector de OSCC.

Para repetir localmente, consultar [PHASE0.md](PHASE0.md).

## Onde seguir

[Roadmap somente de sandbox](roadmap.md) ·
[Análise de mudanças de escopo](docs/roadmap-reassessment.md) ·
[Ambiente Fase 1](docs/phase1-design.md) ·
[Harness de testes Fase 2](docs/phase2-test-harness.md) ·
[Checklist operacional](test-checklist-roadmap.md).

Dados de pacientes, pesos, features derivadas e relatórios
sensíveis permanecem fora do Git, conforme `.gitignore`.
