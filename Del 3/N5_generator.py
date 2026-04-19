from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class SubjGenerator:
    """
    Den här klassen ansvarar för genereringsdelen i RAG-systemet.
    Den tar en fråga, relevant kontext och källor och låter språkmodellen
    formulera ett svar.
    """

    def __init__(
            # Lägger parametrar med standardvärden här, så att det är lätt att gå tillbaka och ändra på bara ett ställe
            self,
            model_name = "AI-Sweden-Models/gpt-sw3-356m-instruct",
            max_contexts=3,         # Har laborerat lite med olika värden, på denna och tokens, och tyckte att detta verkade ok avvägning
            max_new_tokens=150,
            verbose=False
            ):
        """
        Förbereder generatorn men laddar inte modellen direkt.
        Detta minskar risken att kerneln kraschar direkt vid start.
        Modellen laddas först när generate_response() körs. 
        """
     
        # Den språkmodell som ska användas
        self.model_name = model_name

        # Hur många kontextbitar som max ska skickas med i prompten
        self.max_contexts = max_contexts

        # Hur långt svaret max får bli
        self.max_new_tokens = max_new_tokens

        # Om verbose=True skrivs extra debug-info ut
        self.verbose = verbose

        # Här sparas tokenizer, modell och pipeline senare
        self.tokenizer = None
        self.model = None
        self.generator = None

    def _log(self, message):
        # Skriver bara ut debug-info om verbose=True
        if self.verbose:
            print(message)

    def load_model(self):
        """
        Laddar tokenizer, modell och pipeline först när de verkligen behövs.
        """

        # Om modellen redan är laddad behövs inget mer
        if self.generator is not None:
            return
        
        # Laddar tokenizer som omvandlar text till tokens
        # Med try/catch för tydligare felsignal - har haft problem med krashar med hänvisning till åtkomst
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Kunde inte ladda modellen '{self.model_name}'. "
                f"Kontrollera om den finns lokalt eller om du har Hugging Face-access. "
                f"Originalfel: {e}"
            )


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
            device=device,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Hjälp vid felsökning
        self._log(f"Model loaded: {self.model_name}")
        self._log(f"Using device: {device}")

    def _build_context_text(self, contexts, sources):
        # Ser till att vi bara använder lika många källor som kontexter
        paired_items = list(zip(contexts[:self.max_contexts], sources[:self.max_contexts]))

    # Bygger upp källorna som modellen ska läsa
        context_text = ""
        for i, (ctx, src) in enumerate(paired_items, 1):
            context_text += f"KÄLLA {i}: {src}\nTEXT: {ctx}\n\n"

        return context_text    
    
    def _build_prompt(self, query, context_text):
        # Skapar prompten som skickas till modellen
        prompt = f"""Du är en hjälpsam assistent som svarar på frågor om Skolverkets kursplan i matematik.

STRIKTA REGLER (följ EXAKT):
1. Svara ENBART med text som DIREKT står i KÄLLORNA nedan. Inget annat.
2. Kopiera citat från källorna ordagrant när möjligt.
3. Om svaret inte finns i källorna: Svara EXAKT "Jag hittar inte detta i källmaterialet."
4. Svara KORTFATTAT

KÄLLOR:
{context_text}

FRÅGA:
{query}

SVAR:
"""
        return prompt
    
    def _clean_response(self, generated_text, prompt): 
        # Tar bort prompten från svaret om modellen returnerar allt 
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt)].strip()
        else: 
            answer = generated_text.strip()

        # Om modellen inte hittar något svar 
        if not answer: 
            return "Jag hittar inte detta i källmaterialet"
        
        #Stoppar modellen om den börjar upprepa nya etiketter
        stop_markers = ["KÄLLA", "FRÅGA", "SVAR:"]
        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker)[0].strip()

        # Hantering om svaret är tomt efter "städning"
        if not answer: 
            return "Jag hittar inte detta i källmaterialet"
        
        return answer

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
        
        # Bygger upp kontexten som modellen ska läsa
        context_text = self._build_context_text(contexts, sources)

        # Skapar promten som ska skickas
        prompt = self._build_prompt(query, context_text)

        # Hjälp vid felsök
        self._log(f"Question: {query}")
        self._log(f"Number of contexts used: {min(len(contexts), self.max_contexts)}")


        # Kör generering med låg kreativitet 
        result = self.generator(
            prompt,
            max_new_tokens = self.max_new_tokens,      
            do_sample = False,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 3,
            return_full_text=False
        )

        # Tar ut själva genererade texten
        generated_text = result[0]["generated_text"]

        # Städar bort prompt och ev konstiga fortsättningar
        answer = self._clean_response(generated_text, prompt)

        # Hjälp vid felsök 
        self._log(f"Raw generated length: {len(generated_text)}")
        self._log(f"Final answer: {answer}")

        return answer