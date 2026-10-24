"""Character TF-IDF plus reciprocal rank fusion; no additional model call."""

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer


class LexicalIndex:
    def __init__(self, documents: list[Document]):
        self.documents = sorted(documents, key=lambda doc: doc.metadata["chunk_id"])
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True)
        self.matrix = None
        texts = [f"{doc.metadata.get('source', '')}\n{doc.page_content}" for doc in self.documents]
        if any(len(text.strip()) >= 2 for text in texts):
            try:
                self.matrix = self.vectorizer.fit_transform(texts)
            except ValueError as error:
                if "empty vocabulary" not in str(error):
                    raise

    def search(self, query: str, limit: int) -> list[Document]:
        if self.matrix is None or not query.strip():
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], self.documents[index].metadata["chunk_id"]))
        return [self.documents[index] for index in order if scores[index] > 0][:limit]


def reciprocal_rank_fusion(rankings: list[list[Document]], limit: int, constant: int = 60) -> list[Document]:
    if type(limit) is not int or limit < 1 or constant < 1:
        raise ValueError("Positive rank-fusion limits required")
    documents, scores, first_seen = {}, {}, {}
    for ranking in rankings:
        seen = set()
        for rank, document in enumerate(ranking, start=1):
            key = document.metadata["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, document)
            first_seen.setdefault(key, len(first_seen))
            scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))
    return [documents[key] for key in keys[:limit]]
