import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss


class SubjRetriever:
    def __init__(self):
        # Laddar in modell och sparade filer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("data/matematik_index.faiss")
        self.embeddings = np.load("data/matematik_embeddings.npy")
        with open("data/matematik_metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search_chunks(self, query, k=5):
        """
        Hybrid-sök: semantisk sökning + enkel metadata-boost.
        """
        query_lower = query.lower()

        # Sparar ev. hintar från frågan
        year_filter = None
        section_filter = None

        # Regex för årskurs / åk
        year_patterns = [
            r"årskurs\s*(1[0-9]|[1-9])",
            r"åk\s*(1[0-9]|[1-9])",
            r"år\s*(1[0-9]|[1-9])",
            r"klass\s*(1[0-9]|[1-9])",
            r"(1[0-9]|[1-9])\s*-\s*(1[0-9]|[1-9])",
        ]

        # Nyckelord för olika sektioner
        section_patterns = {
            "centralt_innehall": [
                "centralt", "centrala innehållet", "centralt innehåll", "ämnesinnehåll",
                "vad ska man göra", "läras", "lära sig", "innehåll", "lärandemål"
            ],
            "kunskapskrav": [
                "kunskapskrav", "betygskriterier", "betyg", "kriterier", "krav",
                "e-krav", "godkänd", "bedömning", "betygskala", "a-krav",
                "vad ska man klara", "kunna", "klara"
            ],
            "syfte": [
                "syfte", "syftar", "poängen med", "mening", "ändamål",
                "varför", "mål med ämnet", "undervisningens syfte"
            ]
        }

        # Kollar om frågan nämner en årskurs
        for pattern in year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.lastindex == 2:
                    year_filter = f"{match.group(1)}-{match.group(2)}"
                else:
                    year_filter = match.group(1)
                break

        # Kollar om frågan verkar handla om en viss sektion
        for section, keywords in section_patterns.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", query_lower) for keyword in keywords):
                section_filter = section
                break

        # Hjälp vid felsökning
        print(f"Query: '{query}'")
        print(f"Year hint: {year_filter}, Section hint: {section_filter}")

        # Skapar embedding för frågan
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # Söker i FAISS-index
        distances, indices = self.index.search(query_embedding, k * 4)

        # Gör om träffarna till en score-tabell
        faiss_scores = {
            idx: dist for idx, dist in zip(indices[0], distances[0]) if idx != -1
        }

        # Tar fram kandidater till MMR
        candidate_indices = list(faiss_scores.keys())

        # Kör MMR för att få mer varierade träffar
        selected_indices = self.mmr(query_embedding[0], candidate_indices, k * 2)

        results = []
        seen_headings = set()

        for i in selected_indices:
            if i == -1:
                continue

            chunk = self.chunks[i]
            heading = chunk.get("heading", "")

            # Undviker dubbla rubriker i resultatet
            if heading in seen_headings:
                continue

            # Hämtar score från FAISS
            score = faiss_scores.get(i, 0.0)

            # Ger lite extra vikt om årskursen matchar
            if year_filter and str(chunk.get("year", "")) == str(year_filter):
                score *= 1.1

            # Ger lite extra vikt om sektionen matchar
            if section_filter and chunk.get("section") == section_filter:
                score *= 1.1

            results.append((score, i, chunk))
            seen_headings.add(heading)

            # Slutar när vi har tillräckligt många resultat
            if len(results) >= k:
                break

        # Sorterar på score
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def retrieve(self, query, k=5):
        # Kör sökfunktionen
        results = self.search_chunks(query, k=k)

        # Plockar ut bara texten från varje chunk
        contexts = [chunk["text"] for _, _, chunk in results]

        # Bygger källhänvisningar
        sources = [
            f"{chunk.get('code', 'GRGRMAT01')} v{chunk.get('version', 'Okänd')} "
            f"(Årskurs: {chunk.get('year', 'N/A')}, Sektion: {chunk.get('section', 'N/A')})"
            for _, _, chunk in results
        ]

        return contexts, sources

    def mmr(self, query_embedding, candidate_indices, k, lambda_param=0.7):
        # Förbereder listor för urval
        selected = []
        candidates = list(candidate_indices)

        # Räknar likhet mellan fråga och varje kandidat
        sim_to_query = {
            i: float(np.dot(query_embedding, self.embeddings[i]))
            for i in candidates
        }

        while len(selected) < k and candidates:
            best_score = None
            best_idx = None

            for i in candidates:
                # Om inget är valt än finns ingen diversitets-penalty
                if not selected:
                    diversity_penalty = 0.0
                else:
                    # Tar högsta likheten mot redan valda chunkar
                    diversity_penalty = max(
                        float(np.dot(self.embeddings[i], self.embeddings[j]))
                        for j in selected
                    )

                # MMR-balans mellan relevans och variation
                mmr_score = (
                    lambda_param * sim_to_query[i]
                    - (1 - lambda_param) * diversity_penalty
                )

                # Sparar bästa kandidaten
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Lägger till bästa träffen och tar bort den från kandidater
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected