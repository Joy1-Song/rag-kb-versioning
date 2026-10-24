"""Fixed retrieval fixtures: test the program boundary, not model accuracy."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from src.rag_service import RagService


class RagBoundaryTests(unittest.TestCase):
    def service(self, documents):
        service = RagService.__new__(RagService)
        service.vector_store = Mock()
        service.vector_store.retrieve.return_value = documents
        service.chat_model = Mock()
        service.chat_model.invoke.return_value = SimpleNamespace(content="Model answer")
        return service

    def test_empty_retrieval_does_not_call_model(self):
        service = self.service([])
        result = service.ask("Refund requirements?", 4)
        service.chat_model.invoke.assert_not_called()
        self.assertIn("资料中没有明确说明", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_docs"], [])

    def test_whitespace_documents_are_not_evidence(self):
        service = self.service([Document(page_content=" \n\t", metadata={"source": "blank.txt"})])
        result = service.ask("Refund requirements?", 4)
        service.chat_model.invoke.assert_not_called()
        self.assertEqual(result["sources"], [])

    def test_mixed_results_keep_only_nonempty_evidence(self):
        service = self.service([Document(page_content="  ", metadata={"source": "blank.txt"}),
                                Document(page_content="Order number required.", metadata={"source": "policy.txt"})])
        result = service.ask("Refund requirements?", 4)
        self.assertEqual(result["sources"], ["policy.txt"])
        self.assertEqual(len(result["retrieved_docs"]), 1)
        prompt = service.chat_model.invoke.call_args.args[0][1].content
        self.assertNotIn("blank.txt", prompt)
        self.assertIn("Order number required.", prompt)

    def test_nonstring_question_is_rejected_before_retrieval(self):
        for value in (None, 42, ["question"]):
            with self.subTest(value=value):
                service = self.service([])
                with self.assertRaises(TypeError):
                    service.ask(value, 4)
                service.vector_store.retrieve.assert_not_called()

    def test_blank_question_is_rejected_before_retrieval(self):
        service = self.service([])
        with self.assertRaises(ValueError):
            service.ask(" \n\t ", 4)
        service.vector_store.retrieve.assert_not_called()

    def test_invalid_k_is_rejected_before_retrieval(self):
        for value in (0, -1, True, 1.5, "4", None):
            with self.subTest(value=value):
                service = self.service([])
                with self.assertRaises((ValueError, TypeError)):
                    service.ask("Question", value)
                service.vector_store.retrieve.assert_not_called()

    def test_query_is_trimmed(self):
        service = self.service([])
        service.ask("  Question  ", 4)
        service.vector_store.retrieve.assert_called_once_with("Question", top_k=4)

    def test_grounded_request_still_calls_model(self):
        service = self.service([Document(page_content="Order number required.", metadata={"source": "policy.txt"})])
        result = service.ask("Refund requirements?", 4)
        service.chat_model.invoke.assert_called_once()
        self.assertEqual(result["answer"], "Model answer")
        self.assertEqual(result["sources"], ["policy.txt"])

    def test_retrieval_error_is_not_disguised_as_no_evidence(self):
        service = self.service([])
        service.vector_store.retrieve.side_effect = TimeoutError("Injected retrieval failure")
        with self.assertRaises(TimeoutError):
            service.ask("Question", 4)
        service.chat_model.invoke.assert_not_called()

    def test_model_error_is_not_disguised_as_no_evidence(self):
        service = self.service([Document(page_content="Evidence")])
        service.chat_model.invoke.side_effect = TimeoutError("Injected model failure")
        with self.assertRaises(TimeoutError):
            service.ask("Question", 4)


if __name__ == "__main__":
    unittest.main()
