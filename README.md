# Trabalho Prático 1 (NLP)

Resumo curto com onde encontrar as informações neste repositório.

- Modelos utilizados

  - Lista de modelos: veja [`MODELS`](scripts/pipeline.py) em [scripts/pipeline.py](scripts/pipeline.py). Exemplos atualmente configurados:
    - `llama-3.1-8b-instant`
    - `llama-3.3-70b-versatile`
    - `openai/gpt-oss-120b`
- Como rodar os experimentos

  1. Instale dependências:

     ```sh
     pip install -r requirements.txt
     ```

     Arquivo: [requirements.txt](requirements.txt)
  2. Crie um arquivo `.env` na raiz com a chave de API:

     - GROQ_API_KEY=<sua_chave>
  3. Execute o pipeline principal a partir do diretório `scripts`:

     ```sh
     cd scripts
     python pipeline.py
     ```

     - O pipeline chama [`computeMessages.py`](scripts/computeMessages.py) para cada modelo.
     - Após finalizar todas as chamadas, o script ``plotGraphs.py`` é utilizado para plotar os gráficos
     - Alternativamente, rode manualmente um modelo:
       ```sh
       python scripts/computeMessages.py --model llama-3.1-8b-instant
       ```
- Como os dados são salvos

  - As respostas e métricas automáticas são salvas como CSVs em `results/`.
- Métricas calculadas

  - As métricas são computadas em [`calcularMetricas`](scripts/computeMessages.py). Implementações individuais:
    - [`jaccard`](scripts/computeMessages.py) — média do índice de Jaccard entre pares.
    - [`perplexidade`](scripts/computeMessages.py) — pseudo-perplexidade unigram.
    - [`selfBleu`](scripts/computeMessages.py) — BLEU-1 simplificado (self-BLEU).
    - [`distinctN`](scripts/computeMessages.py) — distinct-1 / distinct-2 (proporção de n-gramas únicos).
  - Resultado final armazenado em colunas no CSV gerado.
- Estrutura do repositório

  - [data/messages.json](data/messages.json) — conjunto de mensagens de entrada (exemplos).
  - [scripts/computeMessages.py](scripts/computeMessages.py) — script principal que:
    - monta prompts (`customizePrompt`),
    - chama o LLM (`requestLLMResponse`),
    - faz parsing de saída (`parseRespostas`),
    - calcula métricas (`calcularMetricas`),
    - salva resultados (`salvarRespostas`).
  - [scripts/pipeline.py](scripts/pipeline.py) — orquestra execução para múltiplos modelos (`MODELS`, `run_model`, `main`).
  - [results/](results/) — CSVs com respostas, métricas e gráficos gerados.
- Explicação do pipeline

  - O pipeline (em [scripts/pipeline.py](scripts/pipeline.py)) itera a lista [`MODELS`](scripts/pipeline.py) e para cada modelo executa [`computeMessages.py`](scripts/computeMessages.py).
  - Cada execução de `computeMessages.py`:
    - carrega as mensagens de [data/messages.json](data/messages.json),
    - gera um prompt via [`customizePrompt`](scripts/computeMessages.py) para cada configuração (`baseline`, `soft`, `aggressive`),
    - chama o LLM via API groq,
    - parseia respostas,
    - calcula métricas (`jaccard`, `perplexidade`, `selfBleu`, `distinctN`),
    - salva linhas no CSV por modelo/config.
  - O pipeline inclui delays entre requisições para manejar taxa de requisições.
  - Ao final ele utiliza o script ``plotGraphs.py`` para plotar gráficos dos resultados
