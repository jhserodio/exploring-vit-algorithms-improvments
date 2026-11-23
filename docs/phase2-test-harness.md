# Fase 2 — Especificação detalhada do harness de testes

**Status:** planejada. **Esta branch entrega o instrumento de medição,
não a melhoria a ser medida.**

## Contrato de entrada

O executor futuro recebe uma configuração declarativa como a seguinte:

```json
{
  "schema_version": 1,
  "experiment_id": "synthetic-softmax-parity",
  "fixture_id": "smoke/softmax/large-logits-v1",
  "seed": 12345,
  "baseline": {"kind": "cpu-reference", "revision": "v0"},
  "candidate": {"kind": "mock", "revision": "identity"},
  "tensor": {
    "shape": [4, 128],
    "dtype": "f32",
    "layout": "row-major",
    "device": "cpu"
  },
  "tolerance": {
    "max_abs": 0.0001,
    "relative_l2": 0.0001
  },
  "output": "results/experiments"
}
```

O modelo/checkpoint, origem e hashes devem ser incluídos quando
pertinentes; nenhuma informação de paciente é necessária para
desenvolvimento da ferramenta.

## Tabela de fixtures obrigatórias

| Grupo | Entrada sintética | Invariante |
| --- | --- | --- |
| GEMM | Matriz identidade, matrizes retangulares, FP32 fracionário | `A × I = A`, shape e erro |
| Residual | dois tensores variados e dimensões inválidas | soma e erro de contrato |
| Softmax | logits 0, muito grandes, muito negativos, largura 1/128/1370 | finitude, linhas somam 1, estabilidade |
| LayerNorm | vetor constante, escala/bias, epsilon | média/variância, números finitos |
| GELU | valores simétricos e extremos | saída do oráculo, tolerância |
| Atenção | Q/K/V conhecidos, máscara/padding, várias heads | shape, softmax por linha, tokens |
| Patch | RGB CHW sintético, crop não quadrado | ordenação, geometria |
| DenseFeatures | CLS + patches 37×37, registers fictícios | CLS separado, CHW preservado |
| Falhas | troca de patches, perturbação de 1 valor, NaN/Inf | harness detecta a regressão |
| Dispositivo | CPU/CUDA misturados, CUDA ausente | falha explícita, sem fallback |

**Importante:** o catálogo define os testes para futuros algoritmos.
Ele não exige implementar novos kernels nem modificar o modelo.

## Separação dos executores

- `ReferenceOracle`: matemática CPU, Hugging Face opcional;
- `CandidateExecutor`: interface para uma implementação futura;
- `MockCandidate`: identidade, perturbado e incompatível;
- `Comparator`: shape, dtype, layout, finitude, `max_abs`,
  `relative_l2`, ordem de tokens e erro por camada;
- `Reporter`: registro de input hashes, seed, tolerâncias e resultado.

Uma falha no candidato não deve contaminar as fixtures nem reiniciar
os testes do baseline com outra entrada.

## Espacialidade e OSCC

Os testes de geometria serão criados com imagens e máscaras
**artificiais** e identificadores fictícios. Devem detectar: crop
deslocado, patch invertido, register token confundido com patch,
perda de área marcada, escala físico-espacial ausente e mistura de
slides/pacientes em partitions sintéticas.

Não adicionar datasets reais, anotadores clínicos, treinamento,
segmentadores ou classificadores de câncer nesta branch.

## Testes negativos (gate de qualidade)

1. Baseline e mock-identidade produzem sucesso;
2. Um candidato que troca dois patches **falha**;
3. Um candidato que difere além de tolerância **falha**;
4. Um candidato que produz NaN/Inf **falha**;
5. Um candidato que usa CPU apesar de requisitar CUDA **falha**;
6. Um caso sem hash/seed/shape obrigatório **falha**;
7. Um caso com máscara e tile sintético desalinhados **falha**.

Se qualquer teste negativo passar silenciosamente, o harness não está
pronto para experimentação e a Fase 2 permanece aberta.

## Estrutura proposta

```text
experiments/
  schemas/experiment.schema.json
  fixtures/{math,attention,spatial,negative}/
  configs/
  reports/
motor_inferencia/
  src/experiment/{reference,candidate,comparator,reporter}.rs
scripts/
  check_experiment_report.py
.github/workflows/
  experiment-fixtures.yml
```

Os caminhos são uma proposta para a **Fase 2**, não arquivos
já existentes.

## Critério de aceite

O harness funciona em CPU/CI e com mocks determinísticos,
detecta todas as falhas artificiais, gera JSON válido e
preserva a representação espacial. A GPU real pode ser
adicionada por backend, mas não é pré-requisito para testes
de contrato em CI.

A Fase 2 termina com uma **plataforma de testes**, e não com
uma otimização algorítmica pronta.
