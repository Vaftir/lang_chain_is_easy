import os
from dotenv import load_dotenv
load_dotenv()



from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory




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
# inicializa o promt com o template e a variável de histórico
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"), # Placeholder for message history
        ("human", "{input}")
    ]
)
## inicializa o modelo de linguagem com a chave da API
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4o-mini",
)

# inicialize the chain

chain = prompt | llm # LCA LANG CHAN EXPRESSION LANGUAGE

store = {}
## Função para armazenar o histórico de mensagens
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # SEssion ID é o ID da sessão do usuário CONVERSA UNICA COM ESSE MODELO
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def iniciar_assistente_viagem():
    print("Olá! Eu sou seu assistente de viagens. Como posso ajudar você hoje? \n[DIGITE 'sair' para encerrar a conversa]")

    while True:
        pergunta_usuario = input("\nVocê: ")
        if pergunta_usuario.lower() == "sair":
            print("Assistente: Até logo! Tenha uma ótima viagem!")
            break

        resposta = chain_with_history.invoke(
            {'input': pergunta_usuario},
            config={'configurable':{'session_id':'user123'}}
        )

        print(f"\nAssistente: {resposta.content}")
    

if __name__ == "__main__":
    iniciar_assistente_viagem()
   

