# Fase 1 — Critérios técnicos de provisionamento da sandbox

**Objetivo exclusivo:** preparar Rust, CUDA, containers, smoke tests, CI,
contratos e preflight para receber novos experimentos posteriormente.
A Fase 1 **não** contém implementação de algoritmos científicos nem
exige executar o DINOv2 completo na GPU.

## Matriz de aceitação da configuração

| Gate | Critério | Evidência |
| --- | --- | --- |
| E1 | CPU e GPU em containers separados | `Dockerfile.cpu`, `Dockerfile`, Compose |
| E2 | CUDA Rust fixado, compilação reprodutível | `cuda-kernels/Cargo.toml` + `Cargo.lock`; source CI |
| E3 | Contratos independentes do modelo | `operation_contract.rs`, unit tests |
| E4 | Probes de smoke residual/softmax/GEMM | `cuda-kernels/src/main.rs`; compilação CUDA |
| E5 | Preflight e Compose configuráveis sem GPU | Python preflight/test, `docker compose config` |
| E6 | Diagnóstico na GPU real pronto | `scripts/v1-gpu-acceptance.sh` e `collect_env.sh` |
| E7 | Guardrails de privacidade e documentação | `.gitignore`, README, roadmap |

**Encerramento de E1–E7:** as ferramentas, os testes de configuração e
a CI passam. **Não confundir com `gpu-verified`.** O smoke local na
RTX 5070 Ti é um requisito operacional antes do primeiro experimento
com GPU, que depende do computador do pesquisador.

## Distinção de estados

- `source-ready`: `cargo check --locked` passou em CUDA 13.3;
- `sandbox-config-ready`: mais preflight, tests, Compose e documentação;
- `gpu-verified`: GPU real executou smoke e exportou logs;
- `experiment-ready`: harness de fixtures, ablação, benchmark e relatório
  estão prontos (Fases 2–7).

`gpu-verified` e `experiment-ready` não são afirmações feitas por esta
fase. Não inserir em CI falsos resultados de GPU ou resultados clínicos.

## Ambiente-alvo

```text
Fedora host / RTX 5070 Ti (sm_120)
│
├── CPU: rust:1.90-bookworm
│   ├── reference DINOv2 Base (V0)
│   ├── shape/dtype/device contracts
│   └── CI unit/integration + preflight
│
└── CUDA: nvidia/cuda:13.3.0-devel-ubuntu24.04
    ├── Rust toolchain, cutile=0.5.0
    ├── independent CUDA-source CI (sem GPU)
    └── GPU-only smoke: residual / softmax / GEMM
```

As montagens do Docker Compose usam `:z` por SELinux Fedora, o que pode
relabelar os arquivos do host **mesmo em montagens somente leitura**.
Usar pastas de pesquisa dedicadas e nunca montar dados protegidos de
pacientes apenas para fazer smoke tests.

## Comandos

```sh
python3 scripts/sandbox_preflight.py --json
docker compose --profile gpu config --quiet
docker compose run --build --rm vit-cpu

# Próxima validação no PC real:
bash scripts/v1-gpu-acceptance.sh results/v1-gpu
# Ou Docker + NVIDIA Container Toolkit:
docker compose --profile gpu run --build --rm vit-gpu
```

O programa CPU `motor_inferencia --cuda` continua falhando
explicitamente porque o forward GPU completo não faz parte desta branch.
Os kernels atuais são instrumentos de smoke, não candidatos de pesquisa.

## O que foi transferido para futuras branches

Implementar operadores de atenção otimizados, DINOv2 GPU inteiro,
comparar performance científica de kernels, Hibou-B real, heads
supervisionados, treinamento OSCC, máscaras tumorais e avaliações
clínicas. A presente branch somente estabelece os mecanismos que
permitirão medir essas atividades futuramente.
