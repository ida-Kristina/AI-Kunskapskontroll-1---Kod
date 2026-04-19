import json
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss


class SubjRetriever:
    """
    Den här klassen ansvarar för hämtningsdelen i RAG-systemet.
    Den tar en fråga, söker efter relevanta chunkar i indexet och 
    samlar de texter och källor som bäst matchar frågan.
    """
    def __init__(
        self,
        index_path="data/matematik_index.faiss",
        embeddings_path="data/matematik_embeddings.npy",
        metadata_path="data/matematik_metadata.json",
        model_name="all-MiniLM-L6-v2",
        verbose=False
    ):
        # Sparar sökvägar och inställningar i objektet
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.verbose = verbose

        # Kollar om någon av filerna saknas
        missing_files = [
            path for path in [self.index_path, self.embeddings_path, self.metadata_path]
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                "Följande filer saknas i data-mappen: " + ", ".join(missing_files)
            )

        # Laddar in modell och sparade filer
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(self.index_path)
        self.embeddings = np.load(self.embeddings_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Kollar så att antal embeddings, metadata och index-poster matchar
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"Antal chunkar ({len(self.chunks)}) matchar inte antal embeddings ({len(self.embeddings)})."
            )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"FAISS-index innehåller {self.index.ntotal} poster men metadata innehåller {len(self.chunks)} chunkar."
            )

    def _log(self, message):
        # Skriver bara ut debug-info om verbose=True
        if self.verbose:
            print(message)

    def _extract_filters(self, query):
        # Gör frågan till små bokstäver för enklare matchning
        query_lower = query.lower()

        # Sparar ev. hintar från frågan
        year_filter = None
        section_filter = None

        # Regex för årskurs / åk
        year_patterns = [
            r"\bårskurs\.?\s*([1-9])\b",
            r"\båk\.?\s*([1-9])\b",
            r"\bår\.?\s*([1-9])\b",
            r"\bklass\.?\s*([1-9])\b",
            r"\b([1-9])\s*[-–]\s*([1-9])\b",
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
                "syfte", "syftet", "syftar", "poängen med", "mening", "ändamål",
                "varför", "mål med ämnet"
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

        return year_filter, section_filter

    def _boost_score(self, score, chunk, year_filter=None, section_filter=None):
        # Ger lite extra vikt om årskursen matchar
        if year_filter and str(chunk.get("year", "")) == str(year_filter):
            score *= 1.1

        # Ger lite extra vikt om sektionen matchar
        if section_filter and chunk.get("section") == section_filter:
            score *= 1.1

        return score

    def _format_source(self, chunk):
        # Bygger en lite tydligare källhänvisning
        return (
            f"{chunk.get('code', 'GRGRMAT01')} v{chunk.get('version', 'Okänd')} "
            f"(Årskurs: {chunk.get('year', 'N/A')}, Sektion: {chunk.get('section', 'N/A')})"
        )

    def search_chunks(self, query, k=5):
        """
        Hybrid-sök: semantisk sökning + enkel metadata-boost.
        """

        # Hämtar ev. hintar om årskurs och sektion från frågan
        year_filter, section_filter = self._extract_filters(query)

        # Hjälp vid felsökning
        self._log(f"Query: '{query}'")
        self._log(f"Year hint: {year_filter}, Section hint: {section_filter}")

        # Skapar embedding för frågan
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # Söker i FAISS-index och hämtar lite fler kandidater än vi slutligen vill ha (för att ge oss själva lite marginal)
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

        # Håller koll på vilka chunkar som redan liknar varandra för mycket
        seen_keys = set()

        for i in selected_indices:
            if i == -1:
                continue

            chunk = self.chunks[i]

            # Skapar en nyckel för att undvika för lika dubletter
            dedupe_key = (
                chunk.get("heading", ""),
                chunk.get("year", ""),
                chunk.get("section", "")
            )

            # Undviker dubbla rubriker inom samma typ av innehåll
            if dedupe_key in seen_keys:
                continue

            # Hämtar grundscore från FAISS
            score = faiss_scores.get(i, 0.0)

            # Justerar score lite om metadata matchar frågan
            score = self._boost_score(
                score,
                chunk,
                year_filter=year_filter,
                section_filter=section_filter
            )

            # Sparar träffen
            results.append((score, i, chunk))
            seen_keys.add(dedupe_key)

        # Sorterar på score efter att eventuell boost lagts på
        results.sort(key=lambda x: x[0], reverse=True)

        # Returnerar bara de k bästa träffarna
        return results[:k]

    def retrieve(self, query, k=5, min_score=0.35):
        # Kör sökfunktionen
        results = self.search_chunks(query, k=k)

        # Om inga träffar hittades alls
        if not results:
            return {
                "contexts": [],
                "sources": [],
                "scores": [],
                "best_score": 0.0,
                "has_support": False
            }

        # Plockar ut score-värden från träffarna
        scores = [score for score, _, _ in results]

        # Hämtar bästa träffens score
        best_score = scores[0]

        # Räknar även ut medelvärdet för de tre bästa träffarna
        top_n = min(3, len(scores))
        avg_top_score = sum(scores[:top_n]) / top_n

        # Hjälp vid felsökning
        self._log(f"Top scores: {scores[:3]}")
        self._log(f"Best score: {best_score}")
        self._log(f"Average top-{top_n} score: {avg_top_score}")

        # Stoppar vidare svar om träffen är för svag
        if best_score < min_score:
            return {
                "contexts": [],
                "sources": [],
                "scores": scores,
                "best_score": best_score,
                "has_support": False
            }

        # Plockar ut bara texten från varje chunk
        contexts = [chunk["text"] for _, _, chunk in results]

        # Bygger källhänvisningar
        sources = [
            self._format_source(chunk)
            for _, _, chunk in results
        ]

        return {
            "contexts": contexts,
            "sources": sources,
            "scores": scores,
            "best_score": best_score,
            "has_support": True
        }

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
                # Om inget är valt än finns ingen diversitets-penalty (dvs inget "straff" för att vara för lika)
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
    