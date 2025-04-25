import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Importa as bibliotecas necessárias do LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Define o template do prompt que será usado pelo assistente
template = """
Você é um assistente de viagens útil e amigável. Sua tarefa é ajudar os usuários a planejarem suas viagens. 
Comece perguntando as seguintes informações:
1. Qual é o destino da viagem?
2. Quantas pessoas irão viajar?
3. Qual é o orçamento disponível para a viagem?
4. Qual é a data de partida e retorno?
5. Há preferências específicas, como tipo de acomodação ou atividades?

Seja educado e claro em suas perguntas, e forneça sugestões úteis com base nas respostas do usuário.

Historico de mensagens:
{history}

Entrada do usuário: {input}
"""

# Inicializa o prompt com o template e a variável de histórico
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),  # Mensagem inicial do sistema com o template
        MessagesPlaceholder(variable_name="history"),  # Placeholder para o histórico de mensagens
        ("human", "{input}")  # Entrada do usuário
    ]
)

# Inicializa o modelo de linguagem com a chave da API e configurações
llm = ChatOpenAI(
    temperature=0.7,  # Controla a criatividade das respostas
    model="gpt-4o-mini",  # Modelo de linguagem a ser usado
)

# Combina o prompt e o modelo de linguagem em uma cadeia de execução
chain = prompt | llm

# Dicionário para armazenar o histórico de mensagens por sessão
store = {}

# Função para obter o histórico de mensagens de uma sessão específica
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Verifica se o histórico da sessão já existe, caso contrário, cria um novo
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Combina a cadeia de execução com o histórico de mensagens
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Função para recuperar o histórico
    input_messages_key="input",  # Chave para as mensagens de entrada
    history_messages_key="history",  # Chave para o histórico de mensagens
)

# Função principal para iniciar o assistente de viagens
def iniciar_assistente_viagem():
    print("\nOlá! Eu sou seu assistente de viagens. Como posso ajudar você hoje? \n[DIGITE 'sair' para encerrar a conversa]")

    while True:
        # Recebe a entrada do usuário
        pergunta_usuario = input("\nVocê: ")
        if pergunta_usuario.lower() == "sair":
            # Encerra a conversa se o usuário digitar "sair"
            print("Assistente: Até logo! Tenha uma ótima viagem!")
            break

        # Invoca o modelo com a entrada do usuário e o histórico da sessão
        resposta = chain_with_history.invoke(
            {'input': pergunta_usuario},
            config={'configurable': {'session_id': 'user123'}}  # Identificador único da sessão
        )

        # Exibe a resposta do assistente
        print(f"\nAssistente: {resposta.content}")

# Executa o assistente se o script for executado diretamente
if __name__ == "__main__":
    iniciar_assistente_viagem()


