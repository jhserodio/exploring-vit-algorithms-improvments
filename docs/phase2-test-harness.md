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


## Sequência de implementação e evidências exigidas

Para evitar construir a ferramenta de benchmarking antes do comparador,
a execução da Fase 2 fica dividida em cinco entregas verificáveis,
**todas de infraestrutura**:

| Entrega | Artefato esperado | Teste de aceite |
| --- | --- | --- |
| 2.1 Manifest e catálogo | `experiment.schema.json` + loader e fixtures | Rejeitar versão, shape, seed, hash ou backend inválidos |
| 2.2 Oráculo e executor mock | `ReferenceOracle`, `CandidateExecutor`, identidade e mutações | Input imutável e execução determinística por seed |
| 2.3 Comparador numérico | `Comparator` e tolerâncias versionadas | Igualdade passa; erro injetado, shape/dtype incorreto e NaN falham |
| 2.4 Comparador espacial | Coordenadas e máscaras sintéticas; CLS/register separados | Detectar token trocado, deslocamento, crop que remove marcador e grade errada |
| 2.5 Reporter + CI | JSON versionado, testes de contrato e workflow CPU | Fixture íntegra aprovada e cada caso negativo rejeitado |

**Ordem bloqueante:** 2.1 → 2.2 → 2.3 → 2.4 → 2.5. A criação de
uma otimização real ou de um novo modelo **não** é exigida.

### Contrato numérico do comparador

- Validar `shape`, `dtype`, `layout`, comprimento, identificador do
  backend realmente utilizado, finitude e ordem dos tokens **antes**
  de calcular erros; uma falha de estrutura não pode ser mascarada
  por tolerância generosa.
- Para valores finitos, definir
  `max_abs = max(abs(candidate[i] - reference[i]))` e
  `relative_L2 = ||candidate - reference||₂ / max(||reference||₂, 1e-12)`.
  `max_abs` e `relative_L2` precisam satisfazer seus limites.
- NaN/Inf devem ter política explícita definida no manifest: rejeitar
  por padrão; exceções só para testes negativos que esperem rejeição.
  O comparador não pode reportar `passed=true` com erro não finito.
- A mesma fixture de entrada, seed e metadados devem ser entregues
  ao baseline e ao mock. Testar erro claro ao declarar CUDA e executar
  em CPU, sem fallback automático.
- A comparação espacial precisa preservar índice do CLS,
  `[row,col,channel]` e inversão de coordenadas do crop. Uma
  permutação de patches pode conservar estatísticas agregadas e
  ainda assim **deve falhar**.

### Estrutura de relatório esperada

Cada execução mock deve produzir um JSON com:
`schema_version`, `run_id`, `case_id`, `seed`,
`fixture_sha256`, `baseline_revision`, `candidate_revision`,
`requested_device`, `actual_device`, `shape`, `dtype`,
`max_abs`, `relative_L2`, `tolerance`, `passed`,
`failure_reason` e `git_sha`. O relatório de benchmark com
latências e VRAM é de responsabilidade da Fase 4, não desta.

**Prova final de aceitação:** em um workflow CPU, rodar a suíte
positiva e os sete testes negativos listados acima; os testes
negativos **passam somente quando a falha esperada é detectada**.
Repetir a mesma seed e fixture deve produzir os mesmos campos
determinísticos, ignorando apenas timestamps e IDs de execução.

## Restrições de implementação

Usar apenas fixtures públicas ou inteiramente sintéticas. Não baixar
outros modelos, datasets ou imagens histopatológicas e não implementar
kernels científicos nesta fase. `results/` e tensores derivados ficam
ignorados pelo Git. Não comparar tempo de execução como gate da Fase 2:
benchmarks pertencem à Fase 4.
