

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from indexer import connecter_chroma
from outils_base import OUTILS_SQL

load_dotenv()
# ---------- Outils, prompt et modèle ----------

vector_store = connecter_chroma()


@tool
def chercher_politique(question: str) -> str:
    """Cherche dans les règles de la boutique : livraison, retour, annulation, paiement."""

    docs = vector_store.similarity_search(question, k=2)

    return "\n".join(d.page_content for d in docs)
# ---------- Configuration des outils et du modèle ----------

OUTILS = [chercher_politique, *OUTILS_SQL]

# Outils sensibles : exigent une validation humaine
SENSIBLES = {
    "annuler_commande",
    "creer_retour"
}

PROMPT = (
    "Tu es l'assistant du service client d'une boutique en ligne tunisienne. "
    "Réponds en français, brièvement et poliment. N'invente jamais d'information. "
    "Pour une question sur les règles de la boutique, utilise chercher_politique. "
    "Avant toute annulation ou tout retour : consulte la commande, vérifie la règle, "
    "puis agis seulement si la règle est respectée ; sinon explique pourquoi."
)

# Initialisation du modèle Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
).bind_tools(OUTILS)
# ---------- État du graphe ----------

class Etat(TypedDict):

    # Reducer add_messages : les nouveaux messages sont AJOUTÉS à
    # l'historique
    messages: Annotated[list[AnyMessage], add_messages]

    # Reducer operator.add : les listes renvoyées par les nœuds sont
    # CONCATÉNÉES
    journal: Annotated[list[str], operator.add]

    # Pas de reducer : la nouvelle valeur REMPLACE l'ancienne
    nb_appels_llm: int
def agent(state: Etat):
   reponse = llm.invoke([SystemMessage(PROMPT)] + state["messages"])
   return {"messages": [reponse], "nb_appels_llm":
state.get("nb_appels_llm", 0) + 1}
# ---------- Fonction de routage ----------

def router(state: Etat):

    appels = state["messages"][-1].tool_calls

    if not appels:
        return END

    if any(a["name"] in SENSIBLES for a in appels):
        return "validation"

    return "outils"
# ---------- Nœud de validation humaine ----------

def validation(state: Etat):

    appels = state["messages"][-1].tool_calls

    actions = [
        f"{a['name']} {a['args']}"
        for a in appels
    ]

    decision = interrupt(actions)  # PAUSE : l'état est sauvegardé par le checkpointer

    if decision.strip().lower() in ("oui", "o"):

        return Command(
            goto="outils",
            update={
                "journal": [
                    f"✅ validé : {x}"
                    for x in actions
                ]
            }
        )

    # Refus : chaque appel d'outil doit recevoir une réponse
    refus = [
        ToolMessage(
            "Action refusée par le responsable.",
            tool_call_id=a["id"],
            name=a["name"],
            status="error"
        )
        for a in appels
    ]

    return Command(
        goto="agent",
        update={
            "messages": refus,
            "journal": [
                f"❌ refusé : {x}"
                for x in actions
            ]
        }
    )


# ---------- Graphe ----------
graphe = StateGraph(Etat)
graphe.add_node("agent", agent)
graphe.add_node("validation", validation, destinations=("outils",
"agent"))
graphe.add_node("outils", ToolNode(OUTILS))
graphe.add_edge(START, "agent")
graphe.add_conditional_edges("agent", router, ["outils", "validation",
END])
graphe.add_edge("outils", "agent")
app = graphe.compile(checkpointer=InMemorySaver())
