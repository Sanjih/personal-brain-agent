import os
from openai import OpenAI

def query_personal_brain_agent(user_query: str, retrieved_context: str) -> str:
    """
    Connects to Nebius Token Factory to process user queries using 
    retrieved personal context (RAG pipeline).
    """
    # Nebius Token Factory is OpenAI-compatible
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=os.environ.get("NEBIUS_API_KEY", "your_api_key_here")
    )

    system_prompt = (
        "You are Personal Brain Agent, a privacy-focused productivity assistant. "
        "Use the provided personal context (notes, emails, summaries) to answer "
        "the user's request accurately and concisely."
    )

    user_prompt = f"Context:\n{retrieved_context}\n\nUser Question: {user_query}"

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage / Test stub
    sample_context = "Meeting Notes (2026-08-15): Agreed to launch the new RAG module by next Friday."
    sample_query = "When is the RAG module scheduled to be launched?"
    
    print("--- Personal Brain Agent Test ---")
    print(f"Query: {sample_query}")
    print("Generating response via Nebius Token Factory...")
    # Uncomment line below once NEBIUS_API_KEY is set:
    # print(query_personal_brain_agent(sample_query, sample_context))
