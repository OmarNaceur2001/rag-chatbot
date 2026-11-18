"""
Phase 1 - Ingestion.

Reads PDF, DOCX, and TXT files in data/docs and can crawl one same-origin website from a
starting URL. It chunks the extracted text, creates embeddings, and stores source locations
in ChromaDB.

Usage, from the project root:
    python -m app.ingest                            rebuild the database from data/docs
    python -m app.ingest --search "your question"   show the best chunks (no rebuild)
    python -m app.ingest --url "https://example.com" crawl and index a website
"""
import argparse
import hashlib
import ipaddress
import re
import socket
import sys
import time
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # also lets "python app/ingest.py" work

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from chromadb.errors import NotFoundError  # noqa: E402
from docx import Document  # noqa: E402
from app.gemini_provider import (  # noqa: E402
    GeminiAuthenticationError,
    GeminiConnectionError,
    GeminiError,
    GeminiRateLimitError,
)
from pypdf import PdfReader  # noqa: E402

from app.common import (CHUNK_OVERLAP_WORDS, CHUNK_WORDS, COLLECTION_NAME, DOCS_DIR,  # noqa: E402
                        embed_texts, get_chroma_client, get_collection)


# ======================= 1. Reading the PDFs =======================
def _shape(line: str) -> str:
    """'Page 2 of 4' and 'Page 3 of 4' get the same shape ('Page # of #'), so they count as one line."""
    return re.sub(r"\d+", "#", line.strip())


def remove_headers_footers(pages: list[tuple[int, str]], edge_lines: int = 3) -> list[tuple[int, str]]:
    """Drop running headers, footers and page numbers.

    A line is removed only if it sits in the first/last 3 lines of a page AND the same line
    (digits ignored) shows up there on most pages. Normal content is never touched.
    """
    if len(pages) < 2:
        return pages
    lines_by_page = [(n, [ln.strip() for ln in text.splitlines() if ln.strip()]) for n, text in pages]

    counts: Counter = Counter()
    for _, lines in lines_by_page:
        counts.update({_shape(ln) for ln in lines[:edge_lines] + lines[-edge_lines:]})  # once per page
    min_pages = max(2, round(len(pages) * 0.6))
    repeated = {shape for shape, count in counts.items() if count >= min_pages}

    cleaned = []
    for n, lines in lines_by_page:
        kept = [ln for i, ln in enumerate(lines)
                if not (_shape(ln) in repeated and (i < edge_lines or i >= len(lines) - edge_lines))]
        cleaned.append((n, "\n".join(kept)))
    return cleaned


def load_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for one PDF. Page numbers start at 1."""
    reader = PdfReader(str(pdf_path))
    pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    return remove_headers_footers(pages)


# ======================= 2. Cutting text into chunks =======================
# A sentence ends after . ! ? (but not after a list number like "1.") and a bullet starts a new one.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\b\d\.)\s+|\s+(?=\u2022)")


def clean_text(text: str) -> str:
    """Join the wrapped lines and squeeze the spaces: the page becomes one clean paragraph."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part and part.strip()]


def chunk_text(text: str, max_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Pack whole sentences into chunks of about max_words words.

    Two neighbouring chunks share a few sentences (overlap), so an answer that sits on the
    border between two chunks is still found whole in at least one of them.
    """
    pieces = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for start in range(0, len(words), max_words):      # a very long "sentence" is cut into windows
            pieces.append(" ".join(words[start:start + max_words]))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        n_words = len(piece.split())
        if current and size + n_words > max_words:
            chunks.append(" ".join(current))
            overlap: list[str] = []                        # the last few sentences start the next chunk
            overlap_size = 0
            for previous in reversed(current):
                previous_size = len(previous.split())
                if overlap_size + previous_size > overlap_words:
                    break
                overlap.insert(0, previous)
                overlap_size += previous_size
            current, size = overlap, overlap_size
        current.append(piece)
        size += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_records_for_text(
    text: str,
    source: str,
    title: str,
    page: int | None = None,
    url: str | None = None,
    id_prefix: str | None = None,
) -> list[dict]:
    records = []
    clean = clean_text(text)
    if not clean:
        return records
    prefix = id_prefix or source
    for index, chunk in enumerate(chunk_text(clean)):
        record = {
            "id": f"{prefix}-p{page or 0:03d}-c{index:02d}",
            "text": chunk,
            "source": source,
            "title": title,
            "page": page,
            "chunk": index,
        }
        if url:
            record["url"] = url
        records.append(record)
    return records


def build_records(file_path: Path) -> tuple[list[dict], int]:
    """Extract supported local files into chunks with stable source metadata."""
    title = file_path.stem.replace("_", " ")
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf_pages(file_path)
    elif suffix == ".docx":
        document = Document(str(file_path))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        parts.extend(
            " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            for table in document.tables
            for row in table.rows
        )
        pages = [(None, "\n".join(parts))]
    elif suffix == ".txt":
        pages = [(None, file_path.read_text(encoding="utf-8-sig"))]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    records = []
    for page_number, raw_text in pages:
        records.extend(build_records_for_text(
            raw_text,
            source=file_path.name,
            title=title,
            page=page_number,
            id_prefix=file_path.stem,
        ))
    return records, len(pages)


MAX_CRAWL_PAGES = 500
MAX_CRAWL_DEPTH = 3
MAX_PAGE_BYTES = 2_000_000
CRAWL_TIMEOUT = (3, 10)


def _validated_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP and HTTPS URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("Local network hosts are not allowed.")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(info[4][0])
                for info in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                )
            }
        except socket.gaierror as error:
            raise ValueError(f"Could not resolve website host: {hostname}") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Local and non-public network addresses are not allowed.")
    return urlunparse(parsed._replace(fragment=""))


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port or default_port


def _fetch_web_page(
    url: str,
    origin: tuple[str, str, int],
) -> tuple[str, str, str, list[str]]:
    current_url = url
    for _ in range(6):
        current_url = _validated_url(current_url)
        if _origin(current_url) != origin:
            raise ValueError("Website redirects must remain on the starting origin.")

        with requests.get(
            current_url,
            headers={"User-Agent": "TechNovaRAGBot/1.0"},
            timeout=CRAWL_TIMEOUT,
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError(f"Redirect from {current_url} has no Location header.")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                raise ValueError(f"Not an HTML page: {current_url}")

            body = bytearray()
            for part in response.iter_content(chunk_size=65536):
                body.extend(part)
                if len(body) > MAX_PAGE_BYTES:
                    raise ValueError(f"Web page exceeds the {MAX_PAGE_BYTES}-byte limit: {current_url}")
            html = bytes(body).decode(response.encoding or "utf-8", errors="replace")

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else current_url
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        content = soup.body or soup
        text = content.get_text(" ", strip=True)
        links = [anchor.get("href", "") for anchor in soup.find_all("a", href=True)]
        return current_url, title, text, links

    raise ValueError(f"Too many redirects while fetching {url}")


def crawl_website(
    start_url: str,
    max_pages: int = MAX_CRAWL_PAGES,
    max_depth: int = MAX_CRAWL_DEPTH,
) -> list[dict]:
    """Crawl HTML pages on the starting origin within the configured limits."""
    if not 1 <= max_pages <= MAX_CRAWL_PAGES:
        raise ValueError(f"max_pages must be between 1 and {MAX_CRAWL_PAGES}.")
    if not 0 <= max_depth <= MAX_CRAWL_DEPTH:
        raise ValueError(f"max_depth must be between 0 and {MAX_CRAWL_DEPTH}.")

    start_url = _validated_url(start_url)
    start_origin = _origin(start_url)
    queue = deque([(start_url, 0)])
    scheduled = {start_url}
    records = []
    pages_seen = 0
    while queue and pages_seen < max_pages:
        page_url, depth = queue.popleft()
        try:
            final_url, title, text, links = _fetch_web_page(page_url, start_origin)
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"Could not crawl {page_url}: {error}") from error
        pages_seen += 1
        page_id = hashlib.sha256(final_url.encode("utf-8")).hexdigest()[:20]
        records.extend(build_records_for_text(
            text,
            source=final_url,
            title=title or final_url,
            url=final_url,
            id_prefix=f"web-{page_id}",
        ))

        if depth >= max_depth:
            continue
        for link in links:
            candidate = urljoin(final_url, link)
            parsed_candidate = urlparse(candidate)
            if parsed_candidate.scheme not in {"http", "https"} or not parsed_candidate.hostname:
                continue
            if _origin(candidate) != start_origin:
                continue
            try:
                candidate = _validated_url(candidate)
            except ValueError:
                continue
            if candidate not in scheduled:
                scheduled.add(candidate)
                queue.append((candidate, depth + 1))
    return records


# ======================= 3. Storing in ChromaDB =======================
def create_collection(db):
    """Create an empty collection that ranks by cosine distance (the usual choice for text)."""
    try:
        return db.create_collection(COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})
    except TypeError:                                      # older chromadb versions
        return db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest(web_urls: list[str] | None = None) -> None:
    started = time.time()
    supported_extensions = {".pdf", ".docx", ".txt"}
    local_files = sorted(
        path for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    ) if DOCS_DIR.exists() else []
    web_urls = web_urls or []
    if not local_files and not web_urls:
        sys.exit(f"No PDF, DOCX, or TXT file found in {DOCS_DIR}; provide a website with --url.")

    print(f"Reading {len(local_files)} local document(s) from {DOCS_DIR}")
    records: list[dict] = []
    for file_path in local_files:
        try:
            file_records, n_pages = build_records(file_path)
        except Exception as error:                         # one damaged PDF must not stop the others
            print(f"  ! {file_path.name}: skipped ({type(error).__name__}: {error})")
            continue
        print(f"  - {file_path.name}: {n_pages} page(s) -> {len(file_records)} chunks")
        if not file_records:
            print("    WARNING: no text found in this document.")
        records.extend(file_records)
    for web_url in web_urls:
        web_records = crawl_website(web_url)
        print(f"  - {web_url}: crawled -> {len(web_records)} chunks")
        records.extend(web_records)
    if not records:
        sys.exit("No text could be extracted from the selected sources: nothing to store.")

    # The title is added in front of each chunk before embedding: it helps the search understand
    # what the chunk is about. The stored text stays clean (without the title).
    print(f"Creating embeddings for {len(records)} chunks...")
    vectors = embed_texts(
        [f"{r['title']}\n{r['text']}" for r in records],
        task_type="RETRIEVAL_DOCUMENT",
    )
    if vectors:
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            sys.exit("Gemini returned inconsistent embedding dimensions; the index was not changed.")

    # Embeddings and validation happen before replacing only the Gemini collection.
    # The legacy OpenAI 'docs' collection is never deleted automatically.
    db = get_chroma_client()
    try:
        db.delete_collection(COLLECTION_NAME)              # clean slate -> no duplicates, no stale chunks
    except NotFoundError:
        pass                                               # first run: there was nothing to delete
    collection = create_collection(db)

    for start in range(0, len(records), 500):
        batch = records[start:start + 500]
        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            embeddings=vectors[start:start + 500],
            metadatas=[
                {
                    key: value
                    for key, value in {
                        "source": record["source"],
                        "title": record["title"],
                        "page": record.get("page"),
                        "url": record.get("url"),
                        "chunk": record["chunk"],
                    }.items()
                    if value is not None
                }
                for record in batch
            ],
        )
    print(f"Done: {collection.count()} chunks stored in ChromaDB ({time.time() - started:.1f} s).")


# ======================= 4. Testing the search =======================
def search(question: str, k: int = 4) -> None:
    """Show the k chunks closest in meaning to the question. No LLM involved yet."""
    try:
        collection = get_collection()
    except NotFoundError:
        sys.exit("The database does not exist yet. Run  python -m app.ingest  first.")
    vector = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    result = collection.query(query_embeddings=[vector], n_results=k,
                              include=["documents", "metadatas", "distances"])
    print(f"\nQuestion: {question}\n")
    for rank, (text, meta, distance) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), start=1):
        shown = text if len(text) <= 220 else text[:220] + "..."
        location = f", page {meta['page']}" if meta.get("page") is not None else ""
        print(f"{rank}. {meta['source']}{location}   (similarity {1 - distance:.2f})")
        print(f"   {shown}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or test the document database.")
    parser.add_argument("--search", metavar="QUESTION",
                        help="show the best matching chunks for a question, without rebuilding")
    parser.add_argument("--url", metavar="URL",
                        help="crawl a website from this starting URL")
    args = parser.parse_args()
    try:
        if args.search:
            if args.url:
                parser.error("--search and --url cannot be used together.")
            search(args.search)
        else:
            ingest([args.url] if args.url else [])
    except GeminiAuthenticationError:
        sys.exit("Gemini rejected the API key. Check GEMINI_API_KEY in .env "
                 "(the complete key, no quotes, no spaces).")
    except GeminiRateLimitError as error:
        sys.exit("Gemini refused the request: rate limit or quota exceeded.\n"
                 "Check the Gemini API quota and free-tier limits.\n"
                 f"Details: {error}")
    except GeminiConnectionError:
        sys.exit("Could not reach Gemini. Check your internet connection, VPN or proxy.")
    except GeminiError as error:
        sys.exit(f"The Gemini embedding request failed safely: {error}")


if __name__ == "__main__":
    main()