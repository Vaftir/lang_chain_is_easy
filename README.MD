
# Assistente de Viagens com LangChain

Este é um assistente de viagens interativo desenvolvido com a biblioteca LangChain. Ele ajuda os usuários a planejar suas viagens, fazendo perguntas relevantes e fornecendo sugestões úteis com base nas respostas.

## Objetivo

Um projeto simples para os que querem ter um primeiro contato com essa poderosa biblioteca

## Funcionalidades

- Faz perguntas sobre o destino, número de viajantes, orçamento, datas e preferências.
- Armazena o histórico de mensagens para manter o contexto da conversa.
- Utiliza o modelo de linguagem OpenAI GPT-4 para gerar respostas.

## Pré-requisitos

- Python 3.8 ou superior
- Uma chave de API válida da OpenAI

## Instalação

1. Clone este repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd Langchain
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração

1. Crie um arquivo `.env` na raiz do projeto (se ainda não existir) e adicione as seguintes variáveis de ambiente:
   ```
   OPENAI_API_KEY=<sua_chave_de_api>
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

2. Certifique-se de que o arquivo `.env` está listado no `.gitignore` para evitar o vazamento de informações sensíveis.

3. Opcionalmente, edite o arquivo `.env_example` para incluir um exemplo de configuração:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

## Uso

1. Execute o aplicativo:
   ```bash
   python app.py
   ```

2. Interaja com o assistente digitando suas respostas. Para encerrar a conversa, digite `sair`.

## Estrutura do Projeto

- `app.py`: Código principal do assistente de viagens.
- `.env`: Arquivo de configuração com variáveis de ambiente (não deve ser compartilhado).
- `.env_example`: Exemplo de configuração para referência.
- `requirements.txt`: Lista de dependências do projeto.

## Observações

- Certifique-se de que sua chave de API da OpenAI tem permissões para usar o modelo especificado (`gpt-4o-mini`).
- O histórico de mensagens é armazenado em memória e não é persistido entre execuções.


```
