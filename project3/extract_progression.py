
import re
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# DÉCODAGE DU TEXTE VERTICAL
# ---------------------------------------------------------------------------

def unscramble_vertical(cell: str) -> str:
    """
    Les cellules à texte rotaté (trimestre, séquence, durée) sont extraites par
    pdfplumber caractère/segment par ligne, dans l'ordre inverse de la lecture
    normale. On inverse l'ordre des segments \\n pour reconstituer le texte.

    Exemple vérifié : 'e\\né\\nr\\nu\\nd' -> segments ['e','é','r','u','d']
    -> inversés ['d','u','r','é','e'] -> "durée"
    """
    if not cell:
        return ""
    segments = cell.split("\n")
    return "".join(reversed(segments)).strip()


def looks_like_overlapping_text(cell: str) -> bool:
    """
    Détecte les cellules où deux textes semblent superposés dans le PDF source
    (alternance suspecte de casse/chiffres qui ne correspond à aucun schéma
    de texte rotatif connu). Ex: 'FonctionChap7:\\nnDéUp é1r3i eAnU 1e7t (5...'
    Heuristique : présence de plusieurs motifs de date ou de chiffres isolés
    entremêlés avec des lettres au milieu de mots.
    """
    if not cell:
        return False
    # Plusieurs séquences "chiffre-lettre-chiffre" collées = signe d'overlap
    suspicious = re.findall(r"[A-Za-zÀ-ÿ]\d[A-Za-zÀ-ÿ]\d", cell)
    return len(suspicious) >= 2


def is_valid_semaine(cell: str) -> bool:
    """
    Une cellule 'semaine' correcte ressemble à 'DU 09 AU 13 SEPT 2024' ou,
    pour les semaines à cheval sur deux mois, 'DU 30 SEPT AU 04 OCT 2024'.
    On vérifie la présence des 3 marqueurs structurels (DU, AU, une année
    sur 4 chiffres) plutôt qu'un motif rigide de positions, pour ne pas
    rejeter à tort les semaines à cheval sur deux mois.
    Si absent alors que la cellule n'est pas vide, c'est probablement une
    superposition de texte (cellules qui se chevauchent dans le PDF source).
    """
    if not cell:
        return True  # cellule vide = pas de problème, sera gérée par le forward-fill
    has_du = re.search(r"\bDU\b", cell, re.IGNORECASE)
    has_au = re.search(r"\bAU\b", cell, re.IGNORECASE)
    has_annee = re.search(r"20\d{2}", cell)
    return bool(has_du and has_au and has_annee)


VACANCES_KEYWORDS = ("CONGÉ", "CONGE", "VACANCE")


def find_vacances_text(row: list[str]) -> str | None:
    """Cherche un texte de vacances dans N'IMPORTE QUELLE cellule de la ligne,
    car sa position de colonne varie selon la mise en page du PDF."""
    for cell in row:
        if cell and any(kw in cell.upper() for kw in VACANCES_KEYWORDS):
            return cell.replace("\n", " ").strip()
    return None


# ---------------------------------------------------------------------------
# MAPPING DYNAMIQUE DES COLONNES
# ---------------------------------------------------------------------------

HEADER_KEYWORDS = {
    "trimestre": ["trim"],
    "sequence": ["séq", "seq"],
    "semaine": ["sem"],
    "chapitre": ["contenu"],
    "lecon": ["leçon", "lecon"],
    "duree": ["urée", "uree", "duré"],
    "observation": ["observ"],
}


def detect_column_mapping(header_row: list[str]) -> dict[str, int]:
    """
    Identifie l'index de chaque colonne à partir de la ligne d'en-tête,
    en testant à la fois le texte brut ET sa version décodée (pour 'durée'
    qui est en texte rotatif dans l'en-tête).
    """
    mapping = {}
    for idx, cell in enumerate(header_row):
        if not cell:
            continue
        candidates = [cell.lower(), unscramble_vertical(cell).lower()]
        for field, keywords in HEADER_KEYWORDS.items():
            if field in mapping:
                continue
            if any(kw in c for c in candidates for kw in keywords):
                mapping[field] = idx
    return mapping


# ---------------------------------------------------------------------------
# STRUCTURE DE SORTIE
# ---------------------------------------------------------------------------

@dataclass
class Lecon:
    trimestre_brut: str = ""       # décodé du texte rotatif - informationnel seulement
    trimestre: int | None = None   # déduit de la date - fiable, à utiliser dans l'app
    sequence_brut: str = ""        # décodé - best effort, peut être incomplet
    semaine: str = ""
    chapitre: str = ""
    lecon: str = ""
    duree_h: float | None = None
    observation: str = ""
    type_contenu: str = "cours"    # cours | evaluation | remediation | revision | vacances | travaux_diriges
    confiance: str = "haute"       # haute | basse (nécessite relecture manuelle)


MOIS_TRIMESTRE = {
    # mois -> trimestre, calendrier scolaire camerounais classique
    9: 1, 10: 1, 11: 1, 12: 1,
    1: 2, 2: 2, 3: 2,
    4: 3, 5: 3, 6: 3,
}

MOIS_REGEX = re.compile(
    r"(SEPT|OCT|NOV|D[EÉ]C|JANV|F[EÉ]V|MARS|AVRIL|MAI|JUIN)", re.IGNORECASE
)
MOIS_MAP = {
    "sept": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
    "janv": 1, "fev": 2, "fév": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
}


def deduire_trimestre(semaine: str) -> int | None:
    """Dérive le trimestre depuis la date (fiable) plutôt que du texte rotatif (fragile)."""
    m = MOIS_REGEX.search(semaine or "")
    if not m:
        return None
    key = m.group(1).lower().replace("é", "e")
    for prefix, mois in MOIS_MAP.items():
        if key.startswith(prefix.replace("é", "e")):
            return MOIS_TRIMESTRE.get(mois)
    return None


def parse_duree(cell: str) -> float | None:
    m = re.search(r"(\d+)\s*h", cell or "")
    return float(m.group(1)) if m else None


def classify_type(contenu: str, chapitre: str) -> str:
    text = f"{contenu} {chapitre}".lower()
    if "congé" in text or "vacance" in text:
        return "vacances"
    if "évaluation sommative" in text or "examen blanc" in text:
        return "evaluation"
    if "compte rendu" in text or "remédiation" in text:
        return "remediation"
    if "révision" in text:
        return "revision"
    if "travaux dirigés" in text:
        return "travaux_diriges"
    return "cours"


CHAPITRE_PREFIX = re.compile(r"(chap\s*\d|chapitre)", re.IGNORECASE)


def clean_chapitre(text: str) -> str:
    """
    Certaines cellules mélangent une note administrative et le nom du
    chapitre dans la même cellule PDF, ex :
    'Prise de contact ; prise en main des élèves Chap1 : Nombres complexes'
    On ne garde que le texte à partir de 'Chap' quand ce motif est présent.
    """
    if not text:
        return text
    m = CHAPITRE_PREFIX.search(text)
    return text[m.start():].strip() if m else text


def is_separator_row(row: list[str]) -> bool:
    """Ligne quasi-vide utilisée comme séparateur visuel dans le PDF."""
    return all((c is None or c.strip() == "") for c in row)


# ---------------------------------------------------------------------------
# EXTRACTION PRINCIPALE
# ---------------------------------------------------------------------------

def extract_progression(pdf_path: str) -> list[Lecon]:
    lecons: list[Lecon] = []

    current_trim_brut = ""
    current_seq_brut = ""
    current_semaine = ""
    current_chapitre = ""
    in_vacances = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                mapping = detect_column_mapping(table[0])

                def get(row, field):
                    idx = mapping.get(field)
                    if idx is None or idx >= len(row):
                        return ""
                    return (row[idx] or "").strip()

                for row in table[1:]:  # skip header
                    if is_separator_row(row):
                        continue

                    trim_raw = get(row, "trimestre")
                    seq_raw = get(row, "sequence")
                    sem_raw = get(row, "semaine").replace("\n", " ")
                    chap_raw = get(row, "chapitre").replace("\n", " ")
                    lecon_raw = get(row, "lecon").replace("\n", " ")
                    duree_raw = get(row, "duree")
                    obs_raw = get(row, "observation").replace("\n", " ")

                    # Détection d'un bloc vacances : on cherche dans TOUTES les cellules
                    # de la ligne, car sa position de colonne varie selon la page.
                    vac_text = find_vacances_text(row)
                    if vac_text:
                        lecons.append(Lecon(
                            trimestre_brut=current_trim_brut,
                            trimestre=deduire_trimestre(current_semaine),
                            sequence_brut="",
                            semaine=current_semaine,
                            chapitre="",
                            lecon=vac_text,
                            duree_h=None,
                            observation="",
                            type_contenu="vacances",
                            confiance="haute",
                        ))
                        continue  # ne PAS forward-fill trim/seq/chapitre depuis cette ligne

                    # Validation de la date avant de l'accepter comme référence courante.
                    # Une cellule "semaine" corrompue (superposition de texte dans le PDF
                    # source) ne doit ni écraser la bonne date précédente, ni fausser le
                    # calcul du trimestre.
                    sem_est_corrompue = bool(sem_raw) and not is_valid_semaine(sem_raw)

                    if trim_raw:
                        current_trim_brut = unscramble_vertical(trim_raw)
                    if seq_raw:
                        current_seq_brut = unscramble_vertical(seq_raw)
                    if sem_raw and not sem_est_corrompue:
                        current_semaine = sem_raw
                    if chap_raw and not looks_like_overlapping_text(chap_raw):
                        current_chapitre = clean_chapitre(chap_raw)

                    if not lecon_raw and not obs_raw:
                        continue  # ligne réellement vide de contenu

                    confiance = "basse" if (
                        sem_est_corrompue
                        or looks_like_overlapping_text(chap_raw)
                        or looks_like_overlapping_text(lecon_raw)
                    ) else "haute"

                    item = Lecon(
                        trimestre_brut=current_trim_brut,
                        trimestre=deduire_trimestre(current_semaine),
                        sequence_brut=current_seq_brut,
                        semaine=current_semaine,
                        chapitre=current_chapitre,
                        lecon=lecon_raw,
                        duree_h=parse_duree(duree_raw),
                        observation=obs_raw,
                        type_contenu=classify_type(lecon_raw, current_chapitre),
                        confiance=confiance,
                    )
                    lecons.append(item)

    return lecons


# ---------------------------------------------------------------------------
# 2. CHUNKING POUR LE RAG
# ---------------------------------------------------------------------------

def build_chunk_text(l: Lecon) -> str:
    """
    Texte final qui sera embeddé. On injecte le contexte (chapitre, trimestre)
    DANS le texte pour que la similarité sémantique le capture, en plus des
    métadonnées structurées (filtrage exact).
    """
    return (
        f"Chapitre: {l.chapitre}. "
        f"Leçon: {l.lecon}. "
        f"Période: {l.trimestre}, {l.sequence_brut}, semaine {l.semaine}."
    )


# ---------------------------------------------------------------------------
# 3. STOCKAGE VECTORIEL (Chroma + modèle multilingue FR)
# ---------------------------------------------------------------------------

def store_in_chroma(lecons: list[Lecon], collection_name= 'progression_tle_f_bt', persist_dir: str = "./chroma_db"):
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Modèle multilingue, correct pour le français, léger (~120 Mo)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    texts = [build_chunk_text(l) for l in lecons]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids = [f"lecon_{i}" for i in range(len(lecons))]
    metadatas = [
        {
            "trimestre": l.trimestre,
            "sequence": l.sequence_brut,
            "semaine": l.semaine,
            "chapitre": l.chapitre,
            "duree_h": l.duree_h or 0,
        }
        for l in lecons
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"{len(lecons)} leçons indexées dans la collection '{collection_name}'.")
    return collection
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    

    pdf_path = "./data/progression Tle F&BT.pdf"
    lecons = extract_progression(pdf_path)

    print(f"Extraction terminée : {len(lecons)} leçons extraites.")
    print("Exemple de leçon extraite :", asdict(lecons[0]) if lecons else "Aucune leçon")

    store_in_chroma(lecons, collection_name='progression_tle_f_bt', persist_dir="./chroma_db")






if __name__ == "__main__":
    main()

