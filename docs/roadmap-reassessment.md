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
| Treino de head/classificador e máscaras de câncer | Retirados; testes sintéticos de schema e espacialidade |
| Carregamento/treino de Hibou-B ou Phikon-v2 | Retirados; mock adapters e contratos aqui |
| Auditoria/rotulagem de datasets reais de pacientes | Retirada; validadores e fixtures fictícias aqui |
| Experimentos clínicos, avaliação de sensibilidade real | Retirados; métrica/schema de relatório apenas |
| Agents e orquestradores autônomos | Fora do sandbox inicial |

## O que permanece

O baseline real DINOv2 Base em CPU (V0), necessário como oráculo;
a pilha CUDA Rust e seu smoke (Fase 1); harness de testes
(Fase 2); interfaces, adapters e mocks (Fase 3);
benchmark/medições (Fase 4); schemas e geometrias sintéticas
(Fase 5); reports/proveniência (Fase 6); runbook e
aceitação end-to-end da sandbox (Fase 7).

## Novo critério de Fase 1

**Configuração pronta e Fase 1 encerrada** significa CI, manifests travados,
Dockerfiles, Compose, scripts de preflight e smoke e documentação
válidos. Não equivale a ter executado a RTX 5070 Ti.
O GPU smoke local continua necessário antes do primeiro experimento
GPU; não deve bloquear a conclusão da **configuração de sandbox**.

## Fase 2 exigia mais detalhe?

**Sim.** Foi expandida em quatro blocos: catálogo/fixtures,
oráculo-comparação, espacialidade sintética e CI/report. Um
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
