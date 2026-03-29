# %%
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class SubjGenerator:
    def __init__(self):
        model_name = "AI-Sweden-Models/gpt-sw3-126m"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                  device=0 if device == "cuda" else -1, max_new_tokens=200)
        
    def generate_response(self, query, contexts, sources):
        context_text = "\n\n".join([f"Källa {i+1}: {ctx}\n({src})" 
                                for i, (ctx, src) in enumerate(zip(contexts, sources))])
        
        prompt = f"""Svara på frågan baserat på läroplanerna:

    KONTEXT: {context_text}

    FRÅGA: {query}

    SVAR:"""
        
        result = self.generator(prompt, 
                        do_sample=True, 
                        temperature=0.1, 
                        max_new_tokens=150)
        return result[0]['generated_text'].split("SVAR:")[-1].strip()
        
        pass


