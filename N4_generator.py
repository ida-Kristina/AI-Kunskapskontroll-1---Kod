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
        Förbereder generatorn men laddar inte modellen direkt.
        Detta minskar risken att kerneln kraschar direkt vid start.
        """

        # Den språkmodell som ska användas
        self.model_name = "AI-Sweden-Models/gpt-sw3-356m-instruct"

        # Här sparas tokenizer, modell och pipeline senare
        self.tokenizer = None
        self.model = None
        self.generator = None

    def load_model(self):
        """
        Laddar tokenizer, modell och pipeline först när de verkligen behövs.
        """

        # Om modellen redan är laddad behövs inget mer
        if self.generator is not None:
            return

        # Laddar tokenizer som omvandlar text till tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Laddar själva språkmodellen
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True
        )

        # Kollar om det finns en GPU
        device = 0 if torch.cuda.is_available() else -1

        # Sätter pad token så modellen vet vad den ska använda vid padding
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        

        # Skapar en pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
            pad_token_id=self.tokenizer.eos_token_id
        )    

    def generate_response(self, query, retrieval_result):
        # Om retrievern bedömer att stödet är för svagt
        if not retrieval_result["has_support"]:
            return "Jag hittar inte detta i källmaterialet."

        # Laddar modellen först när den behövs
        self.load_model()

        # Plockar ut kontexter och källor från retrieval-resultatet
        contexts = retrieval_result["contexts"]
        sources = retrieval_result["sources"]

        # Om det inte finns någon kontext att jobba med
        if not contexts:
            return "Jag hittar inte detta i källmaterialet."

        # Ser till att vi bara använder lika många källor som kontexter
        paired_items = list(zip(contexts[:3], sources[:3]))

        # Bygger upp källorna som modellen ska få läsa
        context_text = ""
        for i, (ctx, src) in enumerate(paired_items, 1):
            context_text += f"KÄLLA {i}: {src}\nTEXT: {ctx}\n\n"

        # Skapar prompten som skickas till modellen
        prompt = f"""Du är en hjälpsam assistent som svarar på frågor om Skolverkets kursplan i matematik.

Regler:
- Svara endast med information som uttryckligen står i källorna nedan.
- Använd inte egen bakgrundskunskap.
- Gissa aldrig.
- Om information saknas i källorna, skriv exakt: Jag hittar inte detta i källmaterialet.
- Svara med högst 2 korta meningar.
- Hänvisa till källa i slutet av varje mening.

KÄLLOR:
{context_text}

FRÅGA:
{query}

SVAR:
"""

        result = self.generator(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            return_full_text=False
        )

        # Tar ut själva genererade texten
        svar = result[0]["generated_text"].strip()

        # Om modellen av någon anledning inte gav något svar
        if not svar:
            return "Jag hittar inte detta i källmaterialet."

        return svar