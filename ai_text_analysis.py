import httpx

OLLAMA_MODEL = "llama3.2-vision" 
OLLAMA_BASE_URL = "http://localhost:11434"

async def ai_text_analysis(prompt_type, df_context):
    prompts = {
        'plan': f"You are an experienced data analyst. Suggest a concise data analysis plan (5-7 bullet points) for:\n{df_context}",
        'final': f"Summarize the key insights from this dataset in 5-7 bullet points. Be specific and actionable:\n{df_context}"
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompts.get(prompt_type, prompts['plan']),
                    "stream": False
                }
            )
            result = response.json()
            return result.get("response", "No response generated.")
    except Exception as e:
        return f"Ollama error: {e}"