from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class SubjGenerator:
    """
    Den här klassen ansvarar för genereringsdelen i RAG-systemet.
    Den tar en fråga, relevant kontext och källor och låter språkmodellen
    formulera ett svar.
    """

    def __init__(self):
        """
        Laddar tokenizer, modell och skapar en textgenerator.
        Detta görs en gång när objektet skapas.
        """

        # Den språkmodell som ska användas
        model_name = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"

        # Laddar tokenizer som omvandlar text till tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Laddar själva språkmodellen
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Kollar om det finns en GPU (det gör det inte i mitt fall🙃)
        device = 0 if torch.cuda.is_available() else -1

        # Sätter pad token så modellen vet vad den ska använda vid padding
        model.config.pad_token_id = tokenizer.eos_token_id

        # Skapar en pipeline
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            pad_token_id=tokenizer.eos_token_id
        )

    def generate_response(self, query, contexts, sources):
        # Om det inte finns någon kontext att jobba med
        if not contexts:
            return "Jag hittar ingen relevant information om detta i källmaterialet."

        # Ser till att vi bara använder lika många källor som kontexter
        paired_items = list(zip(contexts[:3], sources[:3]))

        # Bygger upp källorna som modellen ska få läsa
        context_text = ""
        for i, (ctx, src) in enumerate(paired_items, 1):
            context_text += f"KÄLLA {i}: {src}\n{ctx}\n\n"

        # Skapar prompten som skickas till modellen
        prompt = f"""Du är en hjälpsam assistent som svarar på frågor om Skolverkets kursplan i matematik.

        Regler:
        - Svara endast med stöd av källorna nedan.
        - Om information saknas i källorna, skriv exakt: Jag hittar inte detta i källmaterialet.
        - Svara med högst 3 korta meningar.
        - Upprepa inte samma information.
        - Hänvisa till källa i slutet av relevanta meningar.

        KÄLLOR:
        {context_text}

        FRÅGA:
        {query}

        SVAR:
        """
        
        result = self.generator(
            prompt,
            max_new_tokens=170,
            max_length=None,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            return_full_text=False
        )

        # Tar ut själva genererade texten
        svar = result[0]["generated_text"].strip()
        return svar