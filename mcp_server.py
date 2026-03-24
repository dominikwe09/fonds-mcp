from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

# Konstanten
DB_PATH = "/opt/fonds-mcp/db"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Pydantic-Modelle für Input-Validierung
class FondsSuchenInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    anfrage: str = Field(
        ...,
        description="Kundenprofil als Freitext (z.B. 'Kunde, 52 Jahre, 70.000 €, Risikoklasse 2, 15 Jahre Horizont, Altersvorsorge')",
        min_length=10,
        max_length=500
    )
    min_risikoklasse: int = Field(
        default=1,
        description="Mindest-Risikoklasse des Fonds (1=sehr konservativ, 7=sehr spekulativ). Z.B. 3 um nur RK3+ zu bekommen.",
        ge=1,
        le=7
    )
    max_risikoklasse: int = Field(
        default=5,
        description="Maximale Risikoklasse des Fonds (1=sehr konservativ, 7=sehr spekulativ)",
        ge=1,
        le=7
    )
    anzahl_ergebnisse: int = Field(
        default=5,
        description="Anzahl der zurückgegebenen Treffer",
        ge=1,
        le=10
    )

class FondsListeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risikoklasse: Optional[int] = Field(
        default=None,
        description="Filtert nach genauer Risikoklasse. Ohne Angabe werden alle Fonds gelistet.",
        ge=1,
        le=7
    )

# Modell und DB einmalig beim Programmstart laden (nicht pro Verbindung!)
print("Lade Sprachmodell und Datenbank...")
_model = SentenceTransformer(MODEL_NAME)
_db = chromadb.PersistentClient(path=DB_PATH)
_collection = _db.get_or_create_collection("fonds")
print("Bereit.")

# Lifespan: gibt bereits geladene Ressourcen weiter
@asynccontextmanager
async def app_lifespan(server):
    yield {"model": _model, "collection": _collection}

mcp = FastMCP("fonds_mcp", lifespan=app_lifespan, host="127.0.0.1", port=8000)

def format_ergebnis(doc: str, meta: dict) -> str:
    """Formatiert einen Suchtreffer einheitlich."""
    return (
        f"### {meta['name']}\n"
        f"**Risikoklasse:** {meta['risikoklasse']}\n"
        f"**Produktblatt:** {meta['url']}\n\n"
        f"{doc[:500]}\n"
    )

@mcp.tool(
    name="fonds_suchen",
    annotations={
        "title": "Passende Fonds suchen",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def fonds_suchen(params: FondsSuchenInput, ctx: Context) -> str:
    """
    Sucht passende Investmentfonds anhand eines Kundenprofils.
    Durchsucht die Produktinformationsblätter von Union Investment und DZ Bank.

    Args:
        params (FondsSuchenInput): Validierte Eingabe mit:
            - anfrage (str): Kundenprofil-Beschreibung
            - min_risikoklasse (int): Mindest-Risikoklasse 1-7
            - max_risikoklasse (int): Maximale Risikoklasse 1-7
            - anzahl_ergebnisse (int): Gewünschte Trefferzahl

    Returns:
        str: Relevante Auszüge aus den Produktblättern mit Name, Risikoklasse und Link
    """
    await ctx.report_progress(0.2, "Erstelle Suchvektor...")
    embedding = _model.encode(params.anfrage).tolist()

    await ctx.report_progress(0.6, "Durchsuche Produktdatenbank...")

    # Stufe 1: Mehr Kandidaten holen (4× gewünschte Anzahl, mind. 20)
    n_kandidaten = max(20, params.anzahl_ergebnisse * 4)

    where_filter: dict
    if params.min_risikoklasse > 1:
        where_filter = {"$and": [
            {"risikoklasse": {"$gte": int(params.min_risikoklasse)}},
            {"risikoklasse": {"$lte": int(params.max_risikoklasse)}}
        ]}
    else:
        where_filter = {"risikoklasse": {"$lte": int(params.max_risikoklasse)}}

    ergebnisse = _collection.query(
        query_embeddings=[embedding],
        n_results=n_kandidaten,
        where=where_filter
    )

    if not ergebnisse["documents"][0]:
        return "Keine passenden Fonds gefunden. Bitte Risikoklasse oder Anfrage anpassen."

    # Stufe 2: Reranking — exakte RK bevorzugen, dann nach Distanz
    kandidaten = list(zip(
        ergebnisse["documents"][0],
        ergebnisse["metadatas"][0],
        ergebnisse["distances"][0]
    ))

    def score(item):
        _doc, meta, distance = item
        rk = int(meta["risikoklasse"])
        # Penalty für Fonds unterhalb der Ziel-RK (je weiter weg, desto mehr)
        rk_penalty = (params.max_risikoklasse - rk) * 0.1
        return distance + rk_penalty

    # Deduplizieren nach Fondsname (gleicher Fonds kann mehrere Chunks haben)
    seen_names: set = set()
    dedupliziert = []
    for item in sorted(kandidaten, key=score):
        name = item[1]["name"]
        if name not in seen_names:
            seen_names.add(name)
            dedupliziert.append(item)
        if len(dedupliziert) >= params.anzahl_ergebnisse:
            break

    await ctx.report_progress(1.0, "Fertig.")

    treffer = [format_ergebnis(doc, meta) for doc, meta, _dist in dedupliziert]

    # RK-Verteilung für den Berater
    from collections import Counter
    rk_zaehler = Counter(int(meta["risikoklasse"]) for _doc, meta, _dist in dedupliziert)
    rk_info = ", ".join(f"RK{rk} ({n}×)" for rk, n in sorted(rk_zaehler.items()))

    return "\n---\n".join(treffer) + f"\n\n---\n**Gefundene Risikoklassen:** {rk_info}"


@mcp.tool(
    name="fonds_liste",
    annotations={
        "title": "Alle verfügbaren Fonds auflisten",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def fonds_liste(params: FondsListeInput) -> str:
    """
    Listet alle in der Datenbank verfügbaren Fonds auf.

    Args:
        params (FondsListeInput): Optional mit risikoklasse-Filter

    Returns:
        str: Alphabetisch sortierte Liste mit Name und Risikoklasse
    """
    where_filter = {"risikoklasse": int(params.risikoklasse)} if params.risikoklasse else None
    alle = _collection.get(include=["metadatas"], where=where_filter)

    seen = {}
    for meta in alle["metadatas"]:
        name = meta["name"]
        if name not in seen:
            seen[name] = meta["risikoklasse"]

    if not seen:
        return "Keine Fonds in der Datenbank. Bitte zuerst scraper.py ausführen."

    zeilen = sorted([f"- {name} (Risikoklasse {rk})" for name, rk in seen.items()])
    return f"Verfügbare Fonds ({len(zeilen)}):\n" + "\n".join(zeilen)


if __name__ == "__main__":
    # Streamable HTTP – moderner Standard, funktioniert auf allen Geräten
    mcp.run(transport="sse")
