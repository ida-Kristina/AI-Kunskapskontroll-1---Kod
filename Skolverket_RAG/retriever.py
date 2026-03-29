import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss

class SubjRetriever:
    def __init__(self):
        #laddar in mina skapade filer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index("data/matematik_index.faiss")
        self.embeddings = np.load("data/matematik_embeddings.npy", allow_pickle=True)
        with open("data/matematik_metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
    
    def search_chunks(self, query, k=5):
        # Hybrid-sök: metadatafiltrering + embeddings. Först filtrera på årskurs och section (som är viktigast för att hitta rätt), sedan embeddings.
        query_lower = query.lower()
        
# Steg 1: Extrahera årskurs och section från query
        year_filter = None
        section_filter = None
        
        # Regex för åk 
        year_patterns = [
            r'årskurs\s*([1-9]|1[0-9])', 
            r'åk\s*([1-9])',
            r'år\s*([1-9])',
            r'klass\s*([1-9])' 
            r'([1-9])-([1-9])', 
            r'([1-9])'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if '-' in match.group():
                    year_filter = match.group()
                else:
                    year_num = match.group(1)
                    year_filter = year_num
                break
        
        section_patterns = {
            'centralt_innehall': [
                'centralt', 'centrala innehållet', 'centralt innehåll', 'ämnesinnehåll', 'vad ska man göra',
                'vad ska läras', 'vad ska man lära sig', 'innehåll', 'lärandemål', 'taluppfattning', 'algebra', 'geometri'
            ],
            'kunskapskrav': [
                'kunskapskrav', 'betygskriterier', 'betyg', 'kriterier', 'krav', 'E-krav', 
                'godkänd', 'bedömning', 'betygskala', 'A-krav', 'vad ska man klara', 'måste kunna', 'måste veta', 
                'ska man veta', 'ska kunna'
            ],
            'syfte': [
                'syfte', 'syftar', 'poängen med', 'mening', 'ändamål', 'varför', 'mål med ämnet', 'undervisningens syfte'
            ]
        }
        
        for section, keywords in section_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                section_filter = section
                break
        
        print(f"Query: '{query}'")
        print(f"Filtrerar på year: {year_filter}, section: {section_filter}")
        
# Steg 2: Filtrera kandidater
        candidates = []

        for i, chunk in enumerate(self.chunks):
            if year_filter and year_filter not in chunk['year']:  # Skippa om årskurs fel
                continue
            if section_filter and section_filter != chunk['section']:  # Skippa om section fel
                continue

            candidates.append((i, chunk))  # Håll index + chunk
        
        print(f"Hittade {len(candidates)} kandidater efter filtrering")
        
        if not candidates:
            print("Inga kandidater efter filtrering - kör på alla chunks")
            candidates = [(i, chunk) for i, chunk in enumerate(self.chunks)]
        
# Steg 3: Embeddings på kandidater
        if len(candidates) <= k:
            results = candidates
        else:
            # Query embedding
            query_embedding = self.model.encode([query])
            
            # Beräkna avstånd
            distances = []
            for idx, (i, chunk) in enumerate(candidates):
                chunk_emb = self.embeddings[i]              # Chunk-tekst → vektor (från din .npy-fil)
                dist = np.dot(query_embedding, chunk_emb)   # Matematisk likhet från 0-1
                distances.append((dist, i, chunk))
            
            # Sortera på similarity (högst först)
            distances.sort(key=lambda x: x[0], reverse=True)    # Sortera utifrån högst likhet först
            results = distances[:k]                             # K bästa
        
# Steg 4: Ta bort dubletter
        unique_results = []
        seen_headings = set()
        
        for dist, i, chunk in results:
            heading = chunk['heading']
            if heading not in seen_headings:
                unique_results.append((1 - dist, i, chunk))  # Konvertera till distance
                seen_headings.add(heading)
        
        return unique_results[:k]
    pass
    

    # Retrieve-funktion för att få "clean" data att skicka vidare
    def retrieve(self, query, k=5):

        # Kör sökfunktionen
        results = self.search_chunks(query, k=k)

        # Extrahera bara texten, inte metadata, från results till modellen
        contexts = [chunk['text'] for _, _, chunk in results]

        # Bygger källor för hänvisning
        sources = [f"{chunk.get('code', 'GRGRMAT01')} v{chunk.get('version', 'Okänd')} (Årskurs: {chunk.get('year', 'N/A')}, Sektion: {chunk.get('section', 'N/A')})"  
                   for _, _, chunk in results]
        return contexts, sources