from agent_boutique_state import graphe
from outils_base import creer_base

#initialiser la base de données
creer_base()
#Compiler le graphe sans ckeckpointer personalisée
agent = graphe.compile()