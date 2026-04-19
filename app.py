# Jag testade först att köra lösningen som en Streamlit-app. Appen kraschade dock när modellerna skulle laddas, trots att
# beroenden och filer tillslut funkade som förväntat. Min teori är att den generativa modellen blev för tung för Streamlit-miljön att hantera.
# Jag valde därför att köra modellen lokalt istället.


import os
import warnings
import pandas as pd
import streamlit as st

from N4_retriever import SubjRetriever
from N5_generator import SubjGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Matematik-RAG",
    page_icon="📘",
    layout="wide"
)

@st.cache_resource
def load_models():
    retriever = SubjRetriever()
    generator = SubjGenerator()
    return retriever, generator

def ask_rag(query, retriever, generator, k):
    contexts, sources = retriever.retrieve(query, k=k)
    answer = generator.generate_response(query, contexts, sources)
    return answer, contexts, sources

def run_simple_evaluation(test_questions, retriever, generator, k):
    results = []

    for question_data in test_questions:
        question = question_data["question"]
        expected_keywords = question_data["expected_keywords"]

        answer, contexts, sources = ask_rag(question, retriever, generator, k)

        answer_lower = answer.lower()
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0

        source_score = 1 if len(sources) > 0 else 0
        overall_score = round((keyword_score + source_score) / 2, 2)

        results.append({
            "Fråga": question,
            "Svar": answer,
            "Förväntade nyckelord": ", ".join(expected_keywords),
            "Matchade nyckelord": ", ".join(matched_keywords) if matched_keywords else "-",
            "Antal källor": len(sources),
            "Nyckelordspoäng": round(keyword_score, 2),
            "Källpoäng": source_score,
            "Total poäng": overall_score
        })

    return pd.DataFrame(results)


# Session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []


# Ladda modeller

try:
    retriever, generator = load_models()
except Exception as e:
    st.error("Det gick inte att ladda modeller eller datafiler.")
    st.exception(e)
    st.stop()


# Sidhuvud

st.title("📘 Matematik-RAG i Streamlit")
st.write(
    "Denna app låter användaren ställa frågor om matematikens kursplan och få svar "
    "baserade på det källmaterial som har indexerats i systemet."
)


# Sidebar
with st.sidebar:
    st.header("Inställningar")

    k_value = st.slider(
        "Antal hämtade träffar",
        min_value=1,
        max_value=5,
        value=4
    )

    show_context = st.checkbox("Visa hämtad kontext", value=False)
    show_sources = st.checkbox("Visa källor", value=True)

    st.markdown("---")
    st.subheader("Exempelfrågor")
    st.markdown("- Vad är syftet med matematikundervisning?")
    st.markdown("- Vad ska man kunna i geometri i årskurs 6?")
    st.markdown("- Vad är centralt innehåll i algebra för årskurs 4-6?")
    st.markdown("- Vad krävs för E i matte åk 6?")

    st.markdown("---")
    if st.button("Rensa chatten"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_contexts = []
        st.rerun()


# Flikar
tab1, tab2 = st.tabs(["Chatt", "Evaluering"])


# Flik 1: Chatt
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Skriv din fråga här...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Söker i källmaterialet och genererar svar..."):
                try:
                    answer, contexts, sources = ask_rag(
                        query=query,
                        retriever=retriever,
                        generator=generator,
                        k=k_value
                    )

                    st.markdown(answer)

                    st.session_state.last_sources = sources
                    st.session_state.last_contexts = contexts

                    if show_sources:
                        with st.expander("Källor"):
                            if sources:
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Källa {i}:** {source}")
                            else:
                                st.write("Inga källor hittades.")

                    if show_context:
                        with st.expander("Hämtad kontext"):
                            if contexts:
                                for i, ctx in enumerate(contexts, 1):
                                    st.markdown(f"**Kontext {i}:**")
                                    st.write(ctx)
                                    st.markdown("---")
                            else:
                                st.write("Ingen kontext hämtades.")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_text = f"Ett fel uppstod: {e}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text}
                    )

# Flik 2: Evaluering
with tab2:
    st.subheader("Enkel evaluering av chatboten")
    st.write(
        "Här testas modellen på ett antal fasta frågor. "
        "Poängen bygger på om svaret innehåller förväntade nyckelord och om källor hittas."
    )

    test_questions = [
        {
            "question": "Vad är syftet med matematikundervisning?",
            "expected_keywords": ["syfte", "matematik", "undervisning"]
        },
        {
            "question": "Vad krävs för E i matte åk 6?",
            "expected_keywords": ["E", "årskurs", "kunskapskrav"]
        },
        {
            "question": "Vad är centralt innehåll i algebra för årskurs 4-6?",
            "expected_keywords": ["algebra", "centralt innehåll", "4-6"]
        }
    ]

    if st.button("Kör evaluering"):
        with st.spinner("Kör tester..."):
            try:
                eval_df = run_simple_evaluation(
                    test_questions=test_questions,
                    retriever=retriever,
                    generator=generator,
                    k=k_value
                )

                st.dataframe(eval_df, use_container_width=True)

                average_score = round(eval_df["Total poäng"].mean(), 2)
                st.metric("Medelpoäng", average_score)

            except Exception as e:
                st.error("Det gick inte att köra evalueringen.")
                st.exception(e)

# Kör: streamlit run Skol_RAG.py
