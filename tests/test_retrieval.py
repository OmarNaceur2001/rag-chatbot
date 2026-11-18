import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.retrieval import RerankError, _rerank, hybrid_search, reciprocal_rank_fusion, tokenize


class RetrievalTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_combines_and_deduplicates_lists(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])

        self.assertEqual(fused[0][0], "b")
        self.assertEqual(len(fused), 3)

    def test_multilingual_tokenizer_keeps_arabic_and_french_terms(self):
        tokens = tokenize("Échange de garantie; ضمان الأجهزة")

        self.assertIn("garantie", tokens)
        self.assertIn("ضمان", tokens)

    def test_hybrid_search_includes_keyword_only_candidates(self):
        class FakeCollection:
            def count(self):
                return 3

            def query(self, **kwargs):
                assert kwargs["include"] == ["documents", "metadatas", "distances"]
                return {
                    "ids": [["a", "b"]],
                    "documents": [["NovaHub pricing", "General support"]],
                    "metadatas": [[
                        {"source": "pricing.pdf", "title": "Pricing", "page": 1},
                        {"source": "support.pdf", "title": "Support", "page": 1},
                    ]],
                    "distances": [[0.1, 0.2]],
                }

            def get(self, **kwargs):
                assert kwargs["include"] == ["documents", "metadatas"]
                return {
                    "ids": ["a", "b", "c"],
                    "documents": ["NovaHub pricing", "General support", "Return policy 30 days"],
                    "metadatas": [
                        {"source": "pricing.pdf", "title": "Pricing", "page": 1},
                        {"source": "support.pdf", "title": "Support", "page": 1},
                        {"source": "returns.pdf", "title": "Returns", "page": 2},
                    ],
                }

        with patch("app.retrieval.get_collection", return_value=FakeCollection()), patch(
            "app.retrieval.embed_texts", return_value=[[0.0, 0.0]]
        ):
            results = hybrid_search(
                "return policy",
                provider=object(),
                candidate_limit=3,
                top_k=3,
                rerank=False,
            )

        self.assertIn("returns.pdf", [result["source"] for result in results])
        self.assertTrue(all("rrf_score" not in result for result in results))

    def test_reranker_orders_candidates_and_rejects_malformed_ids(self):
        candidates = [
            {"id": "a", "title": "A", "text": "first"},
            {"id": "b", "title": "B", "text": "second"},
        ]
        provider = Mock()
        provider.generate.return_value = {"text": '{"ranked_ids":[1,0]}', "function_calls": []}

        ranked = _rerank("query", candidates, provider, top_k=2)
        self.assertEqual([item["id"] for item in ranked], ["b", "a"])

        provider.generate.return_value = {"text": '{"ranked_ids":[0,0]}', "function_calls": []}
        with self.assertRaises(RerankError):
            _rerank("query", candidates, provider, top_k=2)


if __name__ == "__main__":
    unittest.main()
