"""
Couche 2 : retrouver le bon chapitre/leçon (scope) à partir de la question
de l'élève, puis construire un prompt qui contraint le LLM à rester
dans ce périmètre pour la génération de sa réponse.

Ceci réutilise la collection Chroma peuplée par extract_progression.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer


def get_scope_for_question(
    question_eleve: str,
    classe: str,          # ex: "Terminale F/BT"
    collection_name: str,
    persist_dir: str = "./chroma_db",
    top_k: int = 1,
) -> dict:
    """
    Retrouve la/les leçon(s) de la fiche de progression les plus proches
    sémantiquement de la question posée par l'élève.
    """
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    query_emb = model.encode([question_eleve]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
    )

    return {
        "document": results["documents"][0][0],
        "metadata": results["metadatas"][0][0],
    }


def build_system_prompt(scope: dict, classe: str) -> str:
    """
    Construit le prompt système qui contraint le LLM (Groq/autre) à ne
    répondre que dans le périmètre du chapitre/leçon identifié.
    """
    meta = scope["metadata"]
    return f"""Tu es un tuteur pédagogique pour un élève de {classe}, système éducatif camerounais.

PÉRIMÈTRE STRICT DE LA RÉPONSE (issu de la fiche de progression officielle) :
- Chapitre : {meta['chapitre']}
- Contenu autorisé : {scope['document']}
- Période dans l'année : {meta['trimestre']}, {meta['sequence']}

RÈGLES :
1. Réponds UNIQUEMENT dans ce périmètre. N'introduis aucune notion, théorème ou
   méthode qui ne fait pas partie du contenu autorisé ci-dessus, même si elle
   est mathématiquement liée au sujet.
2. Si l'élève pose une question qui dépasse ce périmètre (notion vue dans un
   autre chapitre ou une autre classe), dis-le lui explicitement et propose
   de reformuler dans le cadre du chapitre en cours.
3. Utilise la terminologie et les notations couramment utilisées dans les
   manuels scolaires camerounais (pas les conventions françaises ou
   anglo-saxonnes si elles diffèrent).
4. Adapte le niveau d'explication à un élève de {classe} (pas plus avancé,
   pas plus simpliste que le programme ne l'exige).
5. Si tu n'es pas certain qu'une notion appartienne strictement à ce
   chapitre selon le programme camerounais, dis-le à l'élève plutôt que
   d'affirmer avec assurance.
"""


# ---------------------------------------------------------------------------
# Exemple d'utilisation dans ton pipeline Django/Groq
# ---------------------------------------------------------------------------

def demo():
    scope = get_scope_for_question(
        question_eleve="Comment calculer le module d'un nombre complexe ?",
        classe="Terminale F/BT",
        collection_name="progression_tle_f_bt",
    )
    system_prompt = build_system_prompt(scope, classe="Terminale F/BT")
    print(system_prompt)

    # -> Ensuite, envoie system_prompt + la question de l'élève à Groq/LLM
    # comme messages[0] = {"role": "system", "content": system_prompt}


if __name__ == "__main__":
    demo()
