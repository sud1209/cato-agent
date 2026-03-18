from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_redis import RedisVectorStore
from app.core.agent.model_factory import get_model
from app.core.config import settings
from app.core.agent.embeddings_factory import get_embeddings

llm = get_model(temperature=0.3)

async def get_objection_response(user_input: str, chat_history: list = []):
    # 1. Similarity Search
    vector_store = RedisVectorStore(
        redis_url=settings.REDIS_URL,
        index_name="cato_objections_index",
        embeddings=get_embeddings()
    )
    docs = vector_store.similarity_search(user_input, k=2)
    
    # 2. Format Examples for FewShotChatMessagePromptTemplate
    examples = [
        {"input": doc.page_content, "output": doc.metadata["answer"]} 
        for doc in docs
    ]

    # 3. Define the Example Template (Role-based)
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 4. Final Chat Prompt including History
    final_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Cato, a specialist at Home.LLC. 
        Your tone is empathetic, professional, and disarming.

        ### CONVERSATIONAL RULES:
        1. BREVITY: Keep your response to 2-3 sentences max. Do not explain everything at once.
        2. NO LISTS: Avoid bullet points or long paragraphs.
        3. FLOW: Answer the user's specific concern, then end with a soft follow-up question.
        4. PERSONA: You are a helpful peer, not a technical manual."""),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="chat_history"), # Inject Redis history
        ("human", "{user_input}"),
    ])

    # 5. Chain and Invoke
    chain = final_prompt | llm
    response = await chain.ainvoke({
        "user_input": user_input,
        "chat_history": chat_history
    })
    
    return response.content