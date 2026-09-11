"""
Small RAG demo: retrieve a specific exercise ("Exercice 1", "Exercice 2", ...)
from a fake exercise document, using Qdrant running fully in-memory.

Embeddings are produced by a real multilingual sentence-transformers model
(paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions) — good for French
content like your progression sheets.
"""

import re

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

VECTOR_SIZE = 384  # paraphrase-multilingual-MiniLM-L12-v2 output dimension

# ---------------------------------------------------------------------------
# 1. Real embedding model
# ---------------------------------------------------------------------------
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# 1b. Extract an explicit exercise number from the query, if the user gave one
# ---------------------------------------------------------------------------
def extract_exercise_number(query: str) -> int | None:
    match = re.search(r"exercice\s*(?:n[°o]?\s*|num[eé]ro\s*)?(\d+)", query.lower())
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# 2. Fake exercise document (what you'd normally parse from a PDF)
# ---------------------------------------------------------------------------
EXERCISES = [
    {
        "numero": 1,
        "titre": "Exercice 1",
        "matiere": "Mathématiques",
        "contenu": (
            "Exercice 1: Résoudre dans R l'équation 2x^2 - 5x + 3 = 0. "
            "Calculer le discriminant puis donner les solutions."
        ),
    },
    {
        "numero": 2,
        "titre": "Exercice 2",
        "matiere": "Mathématiques",
        "contenu": (
            "Exercice 2: Soit la suite (Un) définie par U0 = 1 et "
            "U(n+1) = 2Un + 3. Montrer que (Un) est une suite arithmético-géométrique "
            "et déterminer son terme général."
        ),
    },
    {
        "numero": 3,
        "titre": "Exercice 3",
        "matiere": "Physique",
        "contenu": (
            "Exercice 3: Un mobile est lancé avec une vitesse initiale de 20 m/s "
            "sur un plan incliné. Calculer l'accélération et la distance parcourue "
            "après 4 secondes."
        ),
    },
    {
        "numero": 4,
        "titre": "Exercice 4",
        "matiere": "Physique",
        "contenu": (
            "Exercice 4: Un circuit RLC série est alimenté par une tension "
            "sinusoïdale de fréquence 50 Hz. Déterminer l'impédance totale du circuit "
            "et le déphasage entre tension et courant."
        ),
    },
    {
        "numero": 5,
        "titre": "Exercice 5",
        "matiere": "Physique",
        "contenu": (
            "Exercice 5: Un circuit RLC série est alimenté par une tension "
            "sinusoïdale de fréquence 50 Hz. Déterminer l'impédance totale du circuit "
            "et le déphasage entre tension et courant."
        ),
    },
]




# ---------------------------------------------------------------------------
# 3. Set up Qdrant, fully in-memory (no Docker, nothing persisted to disk)
# ---------------------------------------------------------------------------
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="exercices",
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=ex["numero"],
        vector=embed(ex["contenu"]),
        payload=ex,
    )
    for ex in EXERCISES
]
client.upsert(collection_name="exercices", points=points)


# ---------------------------------------------------------------------------
# 4. Retrieval function — this is the "RAG retrieval" step
#
# Two paths, like a real system should have:
#   a) the user names an exact exercise number -> exact filter, no ambiguity
#   b) the user describes a topic -> semantic vector search
# ---------------------------------------------------------------------------
def retrieve_exercise(user_query: str, top_k: int = 1):
    exercise_number = extract_exercise_number(user_query)

    if exercise_number is not None:
        result = client.query_points(
            collection_name="exercices",
            query=embed(user_query),  # required by the API, but...
            query_filter=Filter(
                must=[FieldCondition(key="numero", match=MatchValue(value=exercise_number))]
            ),
            limit=top_k,
        )
    else:
        result = client.query_points(
            collection_name="exercices",
            query=embed(user_query),
            limit=top_k,
        )

    return result.points


# ---------------------------------------------------------------------------
# 5. Demo: user asks for a specific exercise
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_queries = [
        "donne moi un exercice ",
        "je veux l'exercice 1",
        "peux-tu me donner l'exercice numéro 3 ?",
        "donne moi l'exercice sur le circuit RLC",
        "exercice sur la suite arithmético-géométrique",
    ]

    for q in demo_queries:
        print(f"\nQuestion: {q}")
        hits = retrieve_exercise(q, top_k=3)
        if not hits:
            print("  -> Aucun exercice trouvé.")
            continue
        best = hits[0]
        for p in hits:
            print(f"  -> Exercice {p.payload['numero']}: {p.payload['titre']} ({p.payload['matiere']})")
            print(f"     Contenu: {p.payload['contenu']}")