import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse, urlunparse

import chromadb
from docx import Document

from app.ingest import _fetch_web_page, _validated_url, build_records, crawl_website, ingest


class IngestionTests(unittest.TestCase):
    def test_txt_records_keep_source_without_fake_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("A useful text source.", encoding="utf-8")

            records, page_count = build_records(path)

        self.assertEqual(page_count, 1)
        self.assertEqual(records[0]["source"], "notes.txt")
        self.assertIsNone(records[0]["page"])

    def test_docx_records_include_paragraphs_and_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.docx"
            document = Document()
            document.add_paragraph("Paragraph content.")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Table left"
            table.cell(0, 1).text = "Table right"
            document.save(path)

            records, _ = build_records(path)

        text = " ".join(record["text"] for record in records)
        self.assertIn("Paragraph content.", text)
        self.assertIn("Table left | Table right", text)

    def test_crawler_obeys_page_cap_depth_and_same_origin(self):
        start = "https://docs.example.test/"
        pages = {
            start: ("Home", "Home content", ["/one", "https://elsewhere.test/"]),
            "https://docs.example.test/one": ("One", "One content", ["/two"]),
            "https://docs.example.test/two": ("Two", "Two content", ["/three"]),
            "https://docs.example.test/three": ("Three", "Three content", []),
        }

        def fake_fetch(url, _origin):
            title, text, links = pages[url]
            return url, title, text, links

        def strip_fragment(url):
            return urlunparse(urlparse(url)._replace(fragment=""))

        with patch("app.ingest._validated_url", side_effect=strip_fragment), patch(
            "app.ingest._fetch_web_page", side_effect=fake_fetch
        ) as fetch:
            records = crawl_website(start, max_pages=3, max_depth=3)

        self.assertEqual(fetch.call_count, 3)
        self.assertEqual({record["url"] for record in records}, set(list(pages)[:3]))
        self.assertTrue(all(record["page"] is None for record in records))

    def test_crawler_obeys_depth_limit(self):
        start = "https://docs.example.test/"
        pages = {
            start: ("Home", "Home", ["/one"]),
            "https://docs.example.test/one": ("One", "One", ["/two"]),
            "https://docs.example.test/two": ("Two", "Two", ["/three"]),
            "https://docs.example.test/three": ("Three", "Three", ["/four"]),
            "https://docs.example.test/four": ("Four", "Four", []),
        }

        def fake_fetch(url, _origin):
            title, text, links = pages[url]
            return url, title, text, links

        with patch("app.ingest._validated_url", side_effect=lambda url: url), patch(
            "app.ingest._fetch_web_page", side_effect=fake_fetch
        ) as fetch:
            records = crawl_website(start, max_pages=20, max_depth=3)

        self.assertEqual(fetch.call_count, 4)
        self.assertNotIn("https://docs.example.test/four", {record["url"] for record in records})

    def test_fetcher_extracts_html_text_and_links(self):
        class FakeResponse:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            encoding = "utf-8"
            is_redirect = False
            is_permanent_redirect = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                self.requested_chunk_size = chunk_size
                yield b"<html><title>Guide</title><body>Useful text<script>ignore()</script><a href='/next'>Next</a></body></html>"

        url = "https://docs.example.test/"
        with patch("app.ingest._validated_url", return_value=url), patch(
            "app.ingest.requests.get", return_value=FakeResponse()
        ):
            final_url, title, text, links = _fetch_web_page(url, ("https", "docs.example.test", 443))

        self.assertEqual(final_url, url)
        self.assertEqual(title, "Guide")
        self.assertIn("Useful text", text)
        self.assertNotIn("ignore()", text)
        self.assertEqual(links, ["/next"])

    def test_private_network_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-public"):
            _validated_url("http://127.0.0.1/")

    def test_ingest_stores_source_metadata_without_openai(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "guide.txt").write_text("A stored text source.", encoding="utf-8")
            client = chromadb.Client()

            def fake_embeddings(texts, provider=None, **kwargs):
                return [[0.1, 0.2, 0.3] for _ in texts]

            with patch("app.ingest.DOCS_DIR", docs_dir), patch(
                "app.ingest.get_chroma_client", return_value=client
            ), patch("app.ingest.embed_texts", side_effect=fake_embeddings), contextlib.redirect_stdout(io.StringIO()):
                ingest()

            collection = client.get_collection("docs_gemini")
            stored = collection.get(include=["metadatas"])

        self.assertEqual(collection.count(), 1)
        metadata = stored["metadatas"][0]
        self.assertEqual(metadata["source"], "guide.txt")
        self.assertNotIn("page", metadata)


if __name__ == "__main__":
    unittest.main()
