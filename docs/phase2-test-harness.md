# Fase 2 — Especificação detalhada do harness de testes

**Status:** planejada. **Esta branch entrega o instrumento de medição,
não a melhoria a ser medida.** As **imagens de entrada vêm exclusivamente
do acervo já existente** do projeto; tiles e versões de teste só podem
ser derivados delas. Fixtures numéricas artificiais continuam permitidas
para validar matemática de operadores, mas não imagens ou máscaras inventadas.

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

| Grupo | Entrada para teste | Invariante |
| --- | --- | --- |
| GEMM | Matriz identidade, matrizes retangulares, FP32 fracionário | `A × I = A`, shape e erro |
| Residual | dois tensores variados e dimensões inválidas | soma e erro de contrato |
| Softmax | logits 0, muito grandes, muito negativos, largura 1/128/1370 | finitude, linhas somam 1, estabilidade |
| LayerNorm | vetor constante, escala/bias, epsilon | média/variância, números finitos |
| GELU | valores simétricos e extremos | saída do oráculo, tolerância |
| Atenção | Q/K/V conhecidos, máscara/padding, várias heads | shape, softmax por linha, tokens |
| Patch | RGB CHW extraído de imagem existente, crop não quadrado derivado | ordenação, geometria, origem |
| DenseFeatures | CLS + patches 37×37, registers de teste no tensor | CLS separado, CHW preservado |
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

As imagens para testes espaciais serão **somente as já existentes no
acervo do projeto**, acessadas localmente por `DATA_DIR`. Qualquer
tile, crop, redimensionamento ou variante de teste deve ser derivado
de uma delas e registrar `source_image_sha256`, coordenadas e
transformações. Não gerar imagens artificiais independentes.

Se houver máscaras ou anotações de referência no acervo, usá-las
junto com suas imagens originais e aplicar exatamente as mesmas
transformações geométricas. **Sem máscaras existentes, não criar
ground truth falso:** validar apenas ordem de tokens, alinhamento
geométrico e cobertura do crop. Marcar verificações que dependem
de máscaras como indisponíveis, sem alegar localização de lesões.

Os testes negativos podem adulterar **metadados, coordenadas ou
ordem dos tokens** de uma amostra derivada para garantir que o
harness rejeita essas situações; não alteram rótulos clínicos
nem criam novas imagens. Devem detectar crop deslocado, patch
invertido, register token confundido com patch, escala física
ausente e sobreposição indevida de grupos patient/slide.

Nenhuma imagem ou anotação privada deve entrar no repositório,
nos logs de CI ou nos artefatos enviados ao GitHub. A CI pública
testa os operadores e os contratos de metadados sem exigir
acesso ao acervo. Os testes com imagens existentes rodam
localmente ou em ambiente autorizado com acesso controlado.
Esta branch não treina segmentadores ou classificadores.

## Testes negativos (gate de qualidade)

1. Baseline e mock-identidade produzem sucesso;
2. Um candidato que troca dois patches **falha**;
3. Um candidato que difere além de tolerância **falha**;
4. Um candidato que produz NaN/Inf **falha**;
5. Um candidato que usa CPU apesar de requisitar CUDA **falha**;
6. Um caso sem hash/seed/shape obrigatório **falha**;
7. Um caso derivado de imagem existente com transformação/posição
   incoerente **falha**; se houver máscara anotada correspondente,
   um desalinhamento imagem/máscara **também falha**.

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
| 2.4 Comparador espacial | Tiles/crops derivados do acervo, coordenadas e máscaras existentes quando disponíveis | Detectar token trocado, deslocamento e grade errada; cobertura de região anotada só com ground truth |
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

**Prova final de aceitação:** a CI CPU executa os testes de operações
matemáticas e contratos de metadados, inclusive os negativos. Os testes
espaciais dependentes de imagem usam exclusivamente arquivos existentes
via `DATA_DIR` em ambiente autorizado. Os negativos **passam somente
quando a falha esperada é detectada**. Repetir a mesma imagem, seed e
transformação deve produzir os mesmos campos determinísticos,
ignorando apenas timestamps e IDs de execução.

## Restrições de implementação

Usar **somente as imagens já disponíveis no acervo**, sem geração
independente de imagens, tiles ou máscaras. Transformaçōes de imagens
existentes devem preservar rastreabilidade de origem; apenas os testes
numéricos podem usar vetores/matrizes artificiais. Não baixar novos
datasets nem criar rótulos tumorais; não implementar kernels científicos
nesta fase. `DATA_DIR`, `results/` e tensores derivados permanecem fora
do Git. Não comparar tempo como gate da Fase 2: benchmarks pertencem
à Fase 4.
