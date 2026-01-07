from concurrent.futures import ThreadPoolExecutor
from models.openai_model import openai_response
from models.llama_model import llama_response
from models.geminiai_model import gemini_response

def run_parallel(prompt: str):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "chatGPT": executor.submit(openai_response, prompt),
            "llama": executor.submit(llama_response, prompt),
            "gemini": executor.submit(gemini_response, prompt),
            
        }
        for model,future in futures:
            
            try:
                result = future.result()
                results[model] = result
            except Exception as e:
                results[model] = f"Error: {str(e)}"

    return results
