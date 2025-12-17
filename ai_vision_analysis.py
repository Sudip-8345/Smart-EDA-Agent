import httpx
import base64

OLLAMA_MODEL = "llama3.2-vision"  
OLLAMA_BASE_URL = "http://localhost:11434"

async def ai_vision_analysis(img_paths, max_images=5):
    results = []
    images_to_analyze = img_paths[:max_images]
    
    for title, path in images_to_analyze:
        try:
            with open(path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": f"Analyze this '{title}' data visualization chart. Provide 2-3 key insights in bullet points.",
                        "images": [img_base64],
                        "stream": False
                    }
                )
                result = response.json()
                results.append((title, result.get("response", "No response generated.")))
        except Exception as e:
            results.append((title, f'Error: {e}'))
    
    return results