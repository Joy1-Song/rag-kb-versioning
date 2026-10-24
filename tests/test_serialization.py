import unittest

from langchain_core.documents import Document

from src.rag_service import RagService


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="Order number is required.",
                metadata={"source": "guide.pdf", "page": 3, "chunk_id": "A"},
            ),
            Document(
                page_content="Payment proof is also required.",
                metadata={"source": "guide.pdf", "page": 8, "chunk_id": "B"},
            ),
        ]

    def test_empty_input(self):
        self.assertEqual(RagService._serialize_docs([]), [])

    def test_all_documents_keep_count_order_content_and_id(self):
        result = RagService._serialize_docs(self.documents)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            [item["content"] for item in result],
            [document.page_content for document in self.documents],
        )
        self.assertEqual([item["chunk_id"] for item in result], ["A", "B"])

    def test_file_page_and_display_label_remain_matched(self):
        result = RagService._serialize_docs(self.documents)
        self.assertEqual(len(result), len(self.documents))
        for item, document in zip(result, self.documents):
            with self.subTest(page=document.metadata["page"]):
                self.assertEqual(item["metadata"], document.metadata)
                self.assertEqual(
                    item["source"],
                    f"guide.pdf \u7b2c {document.metadata['page']} \u9875",
                )

    def test_optional_metadata_defaults(self):
        result = RagService._serialize_docs([Document(page_content="Only text.")])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Only text.")
        self.assertEqual(result[0]["source"], "unknown")
        self.assertEqual(result[0]["chunk_id"], "")
        self.assertEqual(result[0]["metadata"], {})


if __name__ == "__main__":
    unittest.main()
