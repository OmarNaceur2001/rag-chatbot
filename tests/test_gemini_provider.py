import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from app.gemini_provider import GeminiProvider


class GeminiProviderTests(unittest.TestCase):
    def test_generate_normalizes_text_and_function_calls(self):
        call = SimpleNamespace(
            id="call-1",
            name="calculate",
            args={"expression": "2 + 2"},
        )
        response = SimpleNamespace(candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(text=None, function_call=call),
                SimpleNamespace(text="Answer", function_call=None),
            ]))
        ])
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.api_key = "test"
        provider.model = "gemini-3.8-flash"
        provider.embedding_model = "gemini-embedding-001"
        provider.max_retries = 1
        provider.timeout_seconds = 30
        provider.client = Mock()
        provider.client.models.generate_content.return_value = response

        result = provider.generate(
            [{"role": "user", "content": "Calculate two plus two"}],
            tools=[{
                "function": {
                    "name": "calculate",
                    "description": "Calculate",
                    "parameters": {"type": "object", "properties": {}},
                }
            }],
        )

        self.assertEqual(result["text"], "Answer")
        self.assertEqual(result["function_calls"][0]["name"], "calculate")

    def test_document_and_query_embeddings_use_distinct_task_types(self):
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.api_key = "test"
        provider.model = "gemini-3.8-flash"
        provider.embedding_model = "gemini-embedding-001"
        provider.max_retries = 1
        provider.timeout_seconds = 30
        provider.client = Mock()
        provider.client.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.3, 0.4])]),
        ]

        documents = provider.embed_documents(["document"])
        query = provider.embed_query("question")

        self.assertEqual(documents, [[0.1, 0.2]])
        self.assertEqual(query, [0.3, 0.4])
        document_config = provider.client.models.embed_content.call_args_list[0].kwargs["config"]
        query_config = provider.client.models.embed_content.call_args_list[1].kwargs["config"]
        self.assertEqual(document_config.task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(query_config.task_type, "RETRIEVAL_QUERY")


if __name__ == "__main__":
    unittest.main()
