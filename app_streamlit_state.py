"""
Interface Streamlit pour l'agent de service client.

L'historique, les actions en attente et le journal sont lus
dans l'état du graphe avec app.get_state().
"""

import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_boutique_state import app
from outils_base import creer_base


st.set_page_config(
    page_title="Service client boutique",
    page_icon="🛒",
)

# ---------- Initialisation ----------

st.cache_resource(creer_base)()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
    },
}

snapshot = app.get_state(config)
etat = snapshot.values
actions = snapshot.interrupts[0].value if snapshot.interrupts else None

# ---------- Barre latérale ----------

with st.sidebar:
    st.header("🛒 Boutique")
    st.caption("Agent LangGraph : RAG + SQL + validation humaine")

    if st.button("Nouvelle conversation", use_container_width=True):
        creer_base()
        st.session_state.clear()
        st.rerun()

    with st.expander("Graphe de l'agent"):
        chemin_graphe = Path(__file__).parent / "graphe_agent.png"
        if chemin_graphe.is_file():
            st.image(str(chemin_graphe))
        else:
            st.caption("Image graphe_agent.png introuvable.")

    st.subheader("État du graphe")
    st.metric("Appels au LLM", etat.get("nb_appels_llm", 0))

    st.markdown("**Journal des validations**")
    st.markdown(
        "\n".join(f"- {ligne}" for ligne in etat.get("journal", []))
        or "_Aucune action sensible_"
    )

    st.subheader("Questions à essayer")
    st.markdown(
        "- Quels sont les délais de livraison ?\n"
        "- Où en est la commande 1002 ?\n"
        "- Annule ma commande 1001\n"
        "- J'ai reçu la commande 1003 mais la souris ne marche pas, "
        "je peux la retourner ?\n"
        "- Je veux retourner la commande 1004, le casque ne me plaît plus."
    )

# ---------- Conversation ----------

st.title("Service client")

for message in etat.get("messages", []):
    if message.type == "human":
        st.chat_message("user").markdown(message.text)
    elif message.type == "ai" and message.text:
        st.chat_message("assistant").markdown(message.text)

# ---------- Validation humaine et saisie ----------

envoi_au_graphe = None

if actions:
    st.markdown(
        "**Validation requise**\n\n"
        + "\n".join(f"- `{action}`" for action in actions)
    )

    col1, col2 = st.columns(2)

    if col1.button("✅ Valider", use_container_width=True):
        envoi_au_graphe = Command(resume="oui")

    if col2.button("❌ Refuser", use_container_width=True):
        envoi_au_graphe = Command(resume="non")

question = st.chat_input("Votre question...", disabled=bool(actions))

if question:
    st.chat_message("user").markdown(question)
    envoi_au_graphe = {"messages": [HumanMessage(content=question)]}

# ---------- Exécution du graphe ----------

if envoi_au_graphe is not None:
    with st.spinner("Traitement en cours…"):
        app.invoke(envoi_au_graphe, config)

    st.rerun()