# Reavaliação do roadmap — branch exclusivamente de sandbox

**Decisão do proprietário (2026-11-23):** `feat/testing-background`
prepara ambiente, estrutura e testes para estudos futuros. Implementação
dos algoritmos, aplicações clínicas e treinamento não são tarefas desta branch.

## Removido do roadmap de sandbox

| Escopo anterior | Decisão |
| --- | --- |
| Forward CUDA completo dos 12 blocos DINOv2 | Retirado; implementação futura |
| Operadores otimizados de atenção/LayerNorm/MLP | Retirados; somente probes de toolchain aqui |
| Conclusões científicas de speedup ou ganho de OSCC | Retiradas; criar apenas comparadores e relatórios |
| Treino de head/classificador e geração de máscaras de câncer | Retirados; usar somente anotações já existentes, se disponíveis, para testar alinhamento |
| Carregamento/treino de Hibou-B ou Phikon-v2 | Retirados; mock adapters e contratos aqui |
| Auditoria clínica e nova rotulagem de pacientes | Retiradas; validar apenas metadados do acervo e transformações derivadas |
| Experimentos clínicos, avaliação de sensibilidade real | Retirados; métrica/schema de relatório apenas |
| Agents e orquestradores autônomos | Fora do sandbox inicial |

## O que permanece

O baseline real DINOv2 Base em CPU (V0), necessário como oráculo;
a pilha CUDA Rust e seu smoke (Fase 1); harness de testes
(Fase 2); interfaces, adapters e mocks (Fase 3);
benchmark/medições (Fase 4); schemas e geometrias derivados das
imagens existentes (Fase 5); reports/proveniência (Fase 6); runbook e
aceitação end-to-end da sandbox (Fase 7).

## Novo critério de Fase 1

**Configuração pronta e Fase 1 encerrada** significa CI, manifests travados,
Dockerfiles, Compose, scripts de preflight e smoke e documentação
válidos. Não equivale a ter executado a RTX 5070 Ti.
O GPU smoke local continua necessário antes do primeiro experimento
GPU; não deve bloquear a conclusão da **configuração de sandbox**.

## Fase 2 exigia mais detalhe?

**Sim.** Foi expandida em quatro eixos de trabalho (fixtures,
oráculo/comparação, espacialidade e CI/report) e cinco entregas
sequenciadas com testes de aceite independentes (2.1–2.5). Um
harness só pode ser considerado pronto se detectar erros
injetados deliberadamente: valores modificados, patches trocados,
falhas de device e dados mal formatados.

## Limites de evidência

O resultado V0 de paridade numérica permanece válido, mas não
demonstra capacidade de detectar câncer; o source-build CUDA
não demonstra runtime real na RTX. Nenhum checkbox de
preparação deve ser interpretado como teste científico de
algoritmo otimizado.

[Roadmap revisado](../roadmap.md) ·
[Fase 1](../PHASE1.md) ·
[Fase 2 detalhada](phase2-test-harness.md)

## Correção da origem das imagens (2026-11-23)

O pesquisador **já possui as imagens** que alimentarão a sandbox. Não
gerar imagens ou máscaras sintéticas independentes para validar o
pipeline espacial. Todo tile, crop, resize e caso espacial deriva de
uma imagem existente e registra sua origem. As máscaras, se necessárias,
devem vir de anotações já disponíveis; quando não houver máscara,
verificar geometria sem alegar localização tumoral. Matrizes numéricas
artificiais permanecem válidas para testes isolados de operadores.


## Inventário ORCA efetivamente presente (2026-11-23)

A branch agora contém **200 imagens RGB8 e 200 máscaras
grayscale8 512×512**, em 100 pares training e 100 testing.
O [workflow de auditoria](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45269827098)
confirmou os 200 pareamentos e a ausência de sobreposição
entre os 101 IDs de caso TCGA inferidos de nomes.

**As 200 máscaras são binárias 0/255, não as três classes
do ORCA descritas originalmente.** Portanto a Fase 2 usa
um contrato de anotação binária com proveniência a confirmar;
não deve inventar a classe de tecido não tumoral.
O preprocess V0 atual exclui aproximadamente 8,78% dos
pixels brancos anotados, exigindo testes de cobertura
e registro espacial. Inventário: [ORCA](orca-dataset-audit.md).

Os PNGs estão atualmente commitados no Git público;
não afirmar que ainda se encontram fora do repositório.
Revisar atribuição e redistribuição antes de criar novas cópias.
