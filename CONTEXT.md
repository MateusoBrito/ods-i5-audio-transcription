# Contexto do Projeto: Componente I5 (Fala — VAD e Transcrição)

## Visão Geral
Este repositório contém a implementação do componente **I5**, pertencente à **Camada 2 (Inferência)** da arquitetura ODS 2026/2[cite: 1]. 
A missão central deste componente é atuar como um pipeline de processamento de áudio que separa a fala humana de ruídos de fundo e a converte em texto estruturado com medidas de confiança[cite: 2]. 

## Consumidores e Dependências
- **Consome de:** P5 (Fonte de áudio em buffers contínuos)[cite: 1, 2].
- **Entrega para:** Camada 4, primariamente o componente A7 (Varejo - áudio, voz e gestos) e A6 (Libras)[cite: 1].
- **Base de Tempo:** Todo áudio processado deve manter estritamente o timestamp original da captura (alinhado com o vídeo via componente B3)[cite: 1, 2].

## Arquitetura Interna e Fluxo de Dados (Pipeline)
Para respeitar as restrições de hardware da Nvidia Jetson Nano (budget de GPU ditado pelo componente P2), o fluxo de processamento segue uma ordem imperativa de filtragem em funil[cite: 2]:

1. **Ingestão (Listener):** Recebe o fluxo de áudio contínuo. Atualmente mockado via leitura de arquivos `.wav` locais para testes.
2. **Front-End de Áudio (Processamento em CPU):**
   - **Filtro de Ruído:** Aplicação de algoritmos leves de redução de ruído ambiente e tratamento de sinal degradado[cite: 2].
   - **VAD (Voice Activity Detection):** Avalia os frames de áudio e detecta a presença de voz[cite: 2]. Corta o fluxo e empacota o áudio útil em *batches*. Descarta silêncio absoluto.
3. **Keyword Spotting (Wake Word):** Filtro intermediário que verifica se o *batch* começa com a palavra de ativação. Se negativo, o processamento encerra aqui.
4. **ASR / Transcrição (Processamento em GPU):** O *batch* filtrado é enviado ao modelo acústico local (ex: Whisper via faster-whisper/TensorRT) para conversão em texto[cite: 1, 2].
5. **Extração de Confiança:** O modelo extrai as probabilidades lógicas (*logprobs*) e devolve uma métrica de certeza da transcrição[cite: 2].
6. **Publicação (Adapter B1/B2):** Montagem do payload JSON final e envio para o barramento.

## Restrições Arquiteturais e Critérios de Aceite
- **Processamento 100% Offline:** É estritamente proibido o uso de APIs de nuvem externas. A transcrição deve ocorrer localmente[cite: 2].
- **Proteção de GPU:** O modelo de inferência pesada só pode ser invocado sobre *batches* previamente validados pelo VAD[cite: 2].
- **Dúvida Explícita:** Em cenários de fala sobreposta ou ruído excessivo, o componente não deve falhar silenciosamente ou inventar transcrições; deve retornar o texto com um índice de confiança baixo para que a aplicação (A7) possa pedir confirmação ao usuário[cite: 2].
- **Contrato de Saída:** O payload final de texto deve conter a identificação da câmera/microfone de origem, o horário exato da ocorrência e a versão do modelo utilizado na inferência (B1/B3)[cite: 1].

## Foco Atual de Desenvolvimento (Sprint 1)
O escopo técnico ativo no momento é o **Front-End de Áudio (VAD e Segmentação)**:
- Implementação da lógica de máquina de estados em Python que acumula quadros de áudio quando a fala inicia e encerra o *batch* de gravação quando um limiar de silêncio (ex: 700ms) é atingido.