# Fase 1 — Sandbox Rust/CUDA: ambiente e testes de infraestrutura

**Status: configuração concluída (2026-11-23).** Esta fase prepara o
ambiente e os smoke tests; **não implementa algoritmos de pesquisa nem
inferência CUDA completa do DINOv2**.

## Entregas de provisionamento

- [x] Containers independentes: Rust CPU e CUDA 13.3 + Rust GPU.
- [x] Docker Compose com serviço CPU padrão e perfil explícito `gpu`.
- [x] Cargo manifests e lockfiles distintos, `cutile = "=0.5.0"`.
- [x] Contratos de operações com validação de shape, dtype, layout, device
  e erro explícito quando CUDA não está disponível.
- [x] Kernels sintéticos de *smoke testing*: residual, softmax e GEMM;
  não são melhorias algorítmicas e não substituem um modelo neural.
- [x] CI CPU e CUDA-source (formatação, testes e compilação).
- [x] Preflight offline, testes de configuração, sintaxe Bash e
  validação Docker Compose integrados à CI.
- [x] Script local para verificação da GPU real e coleta de evidências
  (driver, toolkit, GPU, Rust, Git SHA, resultados).

A fase considera-se **pronta para preparar experimentos** quando o
provisionamento e a CI passam. O resultado `gpu-verified` é **separado**:
depende de rodar o script na RTX 5070 Ti e armazenar os resultados.

## Comandos verificados sem GPU

```sh
python3 scripts/sandbox_preflight.py
python3 -m unittest discover -s scripts -p 'test_sandbox_preflight.py'
bash -n scripts/*.sh
docker compose --profile gpu config --quiet

cd motor_inferencia
cargo test --locked
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
```

A compilação do crate GPU é executada na
[CI de CUDA Rust](.github/workflows/cuda-rust-source.yml), dentro do
container de desenvolvimento CUDA 13.3, sem afirmar execução em GPU.

## Comandos disponíveis para a RTX 5070 Ti (ainda não executados aqui)

```sh
# Requer NVIDIA driver, CUDA e Cargo no host:
bash scripts/v1-gpu-acceptance.sh results/v1-gpu

# Alternativa com NVIDIA Container Toolkit:
docker compose --profile gpu run --build --rm vit-gpu
```

Os scripts não possuem fallback silencioso para CPU. Um sucesso na CI
de compilação **não** demonstra o runtime GPU; registre a verificação
de hardware antes do primeiro experimento GPU.

## Próxima fase

**Fase 2 = harness de testes, fixtures, oráculos e falhas controladas.**
Não inclui implementação de kernels de otimização, treinamento,
diagnóstico ou carregamento de imagens reais de pacientes.

Detalhes: [docs/phase1-design.md](docs/phase1-design.md),
[docs/phase2-test-harness.md](docs/phase2-test-harness.md),
[roadmap.md](roadmap.md).
