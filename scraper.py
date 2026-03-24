"""
Union Investment Fonds-Scraper – API-Version
Lädt die komplette Fondsliste über die offizielle Union-Investment-JSON-API,
lädt Produktinformationsblatt-PDFs herunter, extrahiert Text und indexiert
alles in ChromaDB.

API-Endpunkt (aus pk-fundssearch.js reverse-engineered):
  https://internal.api.union-investment.de/beta/web/funddata/fundsearch
  ?api-version=beta-2.0.0&segment=uip&type=fondssuche
  &api-key=6d5b7ad050e948ce99516c20fbe37425

Liefert 242 Fonds in einem einzigen Request, keine Paginierung.
Jeder Fonds enthält: Name, ISIN, WKN, Risikoklasse (int), PIF-URL.
"""

import asyncio
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz
import httpx
import chromadb
from sentence_transformers import SentenceTransformer

try:
    import pytesseract
    from PIL import Image
    OCR_VERFUEGBAR = True
except ImportError:
    OCR_VERFUEGBAR = False

# ─── Konfiguration ─────────────────────────────────────────────────────────────

DB_PATH = "/opt/fonds-mcp/db"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LOG_FILE = "/var/log/fonds-update.log"
CHUNK_GROESSE = 600
CHUNK_UEBERLAPPUNG = 30
MAX_VERSUCHE = 3
MAX_GLEICHZEITIG = 5          # Parallele PDF-Downloads

FUNDDATA_API = (
    "https://internal.api.union-investment.de/beta/web/funddata/fundsearch"
    "?api-version=beta-2.0.0&segment=uip&type=fondssuche"
    "&api-key=6d5b7ad050e948ce99516c20fbe37425"
)

# Schlüsselwörter zur Validierung als Produktinformationsblatt (nicht KID)
PFLICHT_KEYWORDS = ["Anlagestrategie", "Produktinformation", "Anlageziel", "Fondsinformation"]

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

# ─── API: Fondsliste abrufen ───────────────────────────────────────────────────

async def hole_alle_fonds_von_api(client: httpx.AsyncClient) -> list[dict]:
    """
    Ruft alle Fonds von der Union Investment JSON-API ab.
    Gibt Liste zurück: [{name, isin, wkn, risikoklasse, pif_url}]
    """
    log.info(f"Lade Fondsliste von API...")
    for versuch in range(1, MAX_VERSUCHE + 1):
        try:
            resp = await client.get(FUNDDATA_API, timeout=30)
            resp.raise_for_status()
            daten = resp.json()
            break
        except Exception as e:
            log.warning(f"API-Fehler (Versuch {versuch}/{MAX_VERSUCHE}): {e}")
            if versuch == MAX_VERSUCHE:
                log.error("API nicht erreichbar – Abbruch.")
                return []
            await asyncio.sleep(3)

    fonds_liste = []
    for komponente in daten.get("content", {}).get("container", {}).get("component", []):
        if "result" not in komponente:
            continue
        for gruppe in komponente["result"]:
            for row in gruppe.get("tableRows", []):
                pif_url = next(
                    (oi["link"] for oi in row.get("otherInfos", []) if oi.get("name") == "PIF"),
                    None,
                )
                if not pif_url:
                    continue
                risikoklasse = row.get("riskClass", {}).get("value")
                if not isinstance(risikoklasse, int):
                    continue
                fonds_liste.append({
                    "name": row["fundName"]["value"],
                    "isin": row["isin"]["value"],
                    "wkn": row.get("wkn", {}).get("value", ""),
                    "risikoklasse": risikoklasse,
                    "pif_url": pif_url,
                })

    log.info(f"API: {len(fonds_liste)} Fonds mit Produktinformationsblatt gefunden.")
    return fonds_liste

# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def text_zu_chunks(text: str) -> list[str]:
    woerter = text.split()
    chunks, aktuell = [], []
    for wort in woerter:
        aktuell.append(wort)
        if len(" ".join(aktuell)) >= CHUNK_GROESSE:
            chunks.append(" ".join(aktuell))
            aktuell = aktuell[-CHUNK_UEBERLAPPUNG:]
    if aktuell:
        chunks.append(" ".join(aktuell))
    return chunks


def validiere_produktinformation(text: str) -> bool:
    """Prüft ob das PDF wirklich ein Produktinformationsblatt ist (nicht KID)."""
    return any(kw.lower() in text.lower() for kw in PFLICHT_KEYWORDS)


def extrahiere_text_mit_ocr(pdf_bytes: bytes) -> str:
    """Extrahiert Text aus bildbasierten PDFs via OCR (Tesseract)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    seiten_texte = []
    for seite in doc:
        mat = fitz.Matrix(2.0, 2.0)
        pix = seite.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        seiten_texte.append(pytesseract.image_to_string(img, lang="deu"))
    return " ".join(seiten_texte)


def extrahiere_text(pdf_bytes: bytes) -> str:
    """Extrahiert Text aus PDF — direkt, bei Bilddokumenten mit OCR-Fallback."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = " ".join(seite.get_text() for seite in doc).strip()
    if not text and OCR_VERFUEGBAR:
        log.info("  Kein direkter Text — starte OCR...")
        text = extrahiere_text_mit_ocr(pdf_bytes)
    return text


def hat_sich_geaendert(collection, name: str, neuer_hash: str) -> bool:
    vorhandene = collection.get(where={"name": name}, limit=1)
    if not vorhandene["ids"]:
        return True
    return vorhandene["metadatas"][0].get("hash") != neuer_hash


def loesche_fonds(collection, name: str) -> None:
    alte = collection.get(where={"name": name})
    if alte["ids"]:
        collection.delete(ids=alte["ids"])

# ─── PDF laden ────────────────────────────────────────────────────────────────

async def lade_pdf_url(pif_url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Löst den 302-Redirect der document.itl-API auf und gibt die echte PDF-URL zurück."""
    for versuch in range(1, MAX_VERSUCHE + 1):
        try:
            resp = await client.get(pif_url, follow_redirects=False, timeout=15)
            if resp.status_code in (301, 302, 307, 308):
                return str(resp.headers["location"])
            log.warning(f"  PIF-API: Status {resp.status_code}")
            return None
        except httpx.TimeoutException:
            log.warning(f"  PIF-API Timeout (Versuch {versuch}/{MAX_VERSUCHE})")
        except Exception as e:
            log.warning(f"  PIF-API Fehler (Versuch {versuch}/{MAX_VERSUCHE}): {e}")
    return None


async def lade_pdf_bytes(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    """Lädt eine PDF-Datei mit Retry-Logik."""
    for versuch in range(1, MAX_VERSUCHE + 1):
        try:
            resp = await client.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            log.warning(f"  HTTP {e.response.status_code} (Versuch {versuch}/{MAX_VERSUCHE})")
        except httpx.TimeoutException:
            log.warning(f"  Timeout (Versuch {versuch}/{MAX_VERSUCHE})")
        except Exception as e:
            log.warning(f"  Fehler (Versuch {versuch}/{MAX_VERSUCHE}): {e}")
        if versuch < MAX_VERSUCHE:
            await asyncio.sleep(2)
    return None

# ─── Einzelnen Fonds indexieren ────────────────────────────────────────────────

async def indexiere_fonds(
    fonds: dict,
    model: SentenceTransformer,
    collection,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> str:
    name = fonds["name"]
    isin = fonds["isin"]
    risikoklasse = fonds["risikoklasse"]
    pif_url = fonds["pif_url"]

    async with semaphore:
        log.info(f"Prüfe: {name} ({isin})")

        # Schritt 1: Redirect-URL auflösen
        pdf_url = await lade_pdf_url(pif_url, client)
        if not pdf_url:
            log.error(f"  FEHLER: Keine PDF-URL für {isin}")
            return "fehler"

        # Schritt 2: PDF laden
        pdf_bytes = await lade_pdf_bytes(pdf_url, client)
        if not pdf_bytes:
            log.error(f"  FEHLER: PDF nicht ladbar: {pdf_url}")
            return "fehler"

        # Schritt 3: Hash-Check
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        if not hat_sich_geaendert(collection, name, pdf_hash):
            log.info(f"  Unverändert – überspringe.")
            return "unveraendert"

        # Schritt 4: PDF parsen
        try:
            text = extrahiere_text(pdf_bytes)
            n_seiten = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
        except Exception as e:
            log.error(f"  FEHLER beim PDF-Lesen: {e}")
            return "fehler"

        # Schritt 5: Validierung
        if not validiere_produktinformation(text):
            log.warning(f"  WARNUNG: Kein Produktinformationsblatt-Inhalt – überspringe.")
            return "fehler"

        log.info(f"  RK{risikoklasse} | {n_seiten} Seiten | {pdf_url.split('/')[-1][:40]}")

        # Schritt 6: Alt-Einträge löschen, neu chunken + embedden
        loesche_fonds(collection, name)
        chunks = text_zu_chunks(text)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[f"{name}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "name": name,
                    "isin": isin,
                    "risikoklasse": risikoklasse,
                    "url": pdf_url,
                    "hash": pdf_hash,
                }],
            )
        log.info(f"  OK: {len(chunks)} Chunks gespeichert.")
        return "neu"

# ─── Lokale PDFs indexieren ────────────────────────────────────────────────────

async def indexiere_lokale_pdfs(model: SentenceTransformer, collection) -> dict:
    """
    Liest alle PDFs aus /opt/fonds-mcp/lokale_pdfs/ ein,
    liest Metadaten aus metadaten.json und indexiert sie in ChromaDB.

    Unterschied zu Online-PDFs:
    - Kein Download, direkt vom Dateisystem lesen
    - url-Feld enthält "lokal:<dateiname>" statt https://
    - typ "zertifikat" in Metadaten damit fonds_suchen es unterscheiden kann
    """
    stats = {"neu": 0, "unveraendert": 0, "fehler": 0}
    metadaten_pfad = Path("/opt/fonds-mcp/lokale_pdfs/metadaten.json")
    if not metadaten_pfad.exists():
        log.info("Keine metadaten.json gefunden, überspringe lokale PDFs.")
        return stats

    with open(metadaten_pfad, encoding="utf-8") as f:
        metadaten = json.load(f)

    log.info(f"Lokale PDFs: {len(metadaten)} Einträge in metadaten.json")

    for eintrag in metadaten:
        name = eintrag["name"]
        pdf_pfad = Path("/opt/fonds-mcp/lokale_pdfs") / eintrag["datei"]

        if not pdf_pfad.exists():
            log.warning(f"  WARNUNG: {eintrag['datei']} nicht gefunden, überspringe.")
            stats["fehler"] += 1
            continue

        log.info(f"Prüfe lokal: {name}")

        # Hash-Check
        pdf_bytes = pdf_pfad.read_bytes()
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        if not hat_sich_geaendert(collection, name, pdf_hash):
            log.info(f"  Unverändert – überspringe.")
            stats["unveraendert"] += 1
            continue

        # PDF parsen
        try:
            text = extrahiere_text(pdf_bytes)
            n_seiten = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
        except Exception as e:
            log.error(f"  FEHLER beim PDF-Lesen: {e}")
            stats["fehler"] += 1
            continue

        risikoklasse = int(eintrag["risikoklasse"])
        log.info(f"  RK{risikoklasse} | {n_seiten} Seiten | {eintrag['datei']}")

        # Alt-Einträge löschen, neu chunken + embedden
        loesche_fonds(collection, name)
        chunks = text_zu_chunks(text)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[f"{name}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "name": name,
                    "risikoklasse": risikoklasse,
                    "typ": eintrag.get("typ", "fonds"),
                    "emittent": eintrag.get("emittent", "Union Investment"),
                    "url": f"lokal:{eintrag['datei']}",
                    "hash": pdf_hash,
                }],
            )
        log.info(f"  OK: {len(chunks)} Chunks gespeichert.")
        stats["neu"] += 1

    return stats

# ─── Hauptprogramm ────────────────────────────────────────────────────────────

async def indexiere_union_investment(model: SentenceTransformer, collection) -> dict:
    stats = {"neu": 0, "unveraendert": 0, "fehler": 0}
    async with httpx.AsyncClient(
        headers={"User-Agent": "FondsMCP-Scraper/3.0"},
        follow_redirects=False,
    ) as client:
        fonds_liste = await hole_alle_fonds_von_api(client)
        if not fonds_liste:
            log.error("Keine Fonds von API erhalten – überspringe Union Investment.")
            return stats

        log.info(f"Indexiere {len(fonds_liste)} Fonds (max. {MAX_GLEICHZEITIG} parallel)...")
        semaphore = asyncio.Semaphore(MAX_GLEICHZEITIG)
        aufgaben = [
            indexiere_fonds(fonds, model, collection, client, semaphore)
            for fonds in fonds_liste
        ]
        ergebnisse = await asyncio.gather(*aufgaben)

    for e in ergebnisse:
        stats[e] += 1
    return stats


async def indexiere_alle():
    setup_logging()
    start = datetime.now()
    log.info("=" * 60)
    log.info("Fonds-Update gestartet (API-Version)")

    log.info("Lade Sprachmodell...")
    model = SentenceTransformer(MODEL_NAME)
    db = chromadb.PersistentClient(path=DB_PATH)
    collection = db.get_or_create_collection("fonds")

    stats_ui = await indexiere_union_investment(model, collection)
    stats_lokal = await indexiere_lokale_pdfs(model, collection)

    # Gesamtstatistik
    stats = {k: stats_ui[k] + stats_lokal[k] for k in stats_ui}

    dauer = (datetime.now() - start).seconds
    log.info("-" * 60)
    log.info(
        f"Fertig in {dauer}s — "
        f"Neu: {stats['neu']}, Unverändert: {stats['unveraendert']}, Fehler: {stats['fehler']}"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(indexiere_alle())
