# Jag testade först att köra lösningen som en Streamlit-app. Appen kraschade dock när modellerna skulle laddas, trots att
# beroenden och filer tillslut funkade som förväntat. Min teori är att den generativa modellen blev för tung för Streamlit-miljön att hantera.
# Jag valde därför att köra modellen lokalt istället.


import os
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Matematik-RAG", page_icon="📘", layout="wide")

st.title("📘 Matematik-RAG i Streamlit")
st.write("Den här appen låter dig ställa frågor till min RAG-modell med data hämtat från grundskolans kursplaner i matematik.")


@st.cache_resource
def load_models(top_k, min_score):
    from N4_retriever import SubjRetriever
    from N5_generator import SubjGenerator

    retriever = SubjRetriever(verbose=False)
    generator = SubjGenerator(verbose=False)
    return retriever, generator, top_k, min_score

with st.sidebar:
    st.header("Inställningar")
    top_k = st.slider("Antal träffar (top_k)", min_value=1, max_value=5, value=3, step=1)
    min_score = st.slider("Minsta score", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    show_context = st.checkbox("Visa hämtade källor och kontext", value=True)

question = st.text_area(
    "Skriv din fråga här",
    placeholder="Till exempel: Vad är syftet med matematikundervisning?",
    height=120,
)

if st.button("Kör modellen", type="primary"):
    if not question.strip():
        st.warning("Skriv en fråga först.")
    else:
        try:
            with st.spinner("Laddar modell och hämtar relevanta källor..."):
                retriever, generator, top_k, min_score = load_models(top_k, min_score)
                result = retriever.retrieve(question, k=top_k, minscore=min_score)

            st.subheader("Svar")
            if not result.get("hassupport", False):
                st.info("Jag hittar inte detta i källmaterialet.")
            else:
                with st.spinner("Genererar svar..."):
                    answer = generator.generateresponse(question, result)
                st.success(answer)

            if show_context:
                st.subheader("Retrieval-detaljer")
                st.write(f"**Best score:** {result.get('bestscore', 0):.4f}")
                scores = result.get("scores", [])
                if scores:
                    st.write("**Alla scores:**", ", ".join(f"{s:.4f}" for s in scores))

                sources = result.get("sources", [])
                contexts = result.get("contexts", [])

                for i, (src, ctx) in enumerate(zip(sources, contexts), start=1):
                    with st.expander(f"Källa {i}"):
                        st.write(src)
                        st.write(ctx)

        except ModuleNotFoundError as e:
            st.error(
                "Import misslyckades. Kontrollera att du har lagt in `retriever.py` och `generator.py` i samma mapp som appen. "
                f"Detalj: {e}"
            )
        except FileNotFoundError as e:
            st.error(
                "Någon datafil saknas. Kontrollera att din `data/`-mapp innehåller index-, embedding- och metadatafilerna. "
                f"Detalj: {e}"
            )
        except Exception as e:
            st.error(f"Ett oväntat fel uppstod: {e}")

st.markdown("---")
st.caption("Tips: starta appen med kommandot `streamlit run app.py`")

# Kör: streamlit run Skol_RAG.py
