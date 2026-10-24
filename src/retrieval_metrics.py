"""Document-level ranking metrics; empty gold sets do not prove refusal quality."""


def ranking_metrics(ranking: list[str], relevant: list[str], k: int) -> dict:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be positive")
    if len(ranking) != len(set(ranking)) or len(relevant) != len(set(relevant)):
        raise ValueError("Duplicate document IDs would corrupt ranking metrics")
    if not relevant:
        return {"answerable": False, "recall": None, "hit": None, "reciprocal_rank": None,
                "returned_candidates": len(ranking[:k])}
    candidates, gold = ranking[:k], set(relevant)
    hits = gold.intersection(candidates)
    first = next((1 / rank for rank, key in enumerate(candidates, 1) if key in gold), 0.0)
    return {"answerable": True, "recall": len(hits) / len(gold), "hit": int(bool(hits)),
            "reciprocal_rank": first, "returned_candidates": len(candidates)}


def summarize_rankings(records: list[dict], ranking_key: str, k: int) -> dict:
    scored = [ranking_metrics(row[ranking_key], row["relevant_ids"], k) for row in records]
    answerable = [row for row in scored if row["answerable"]]
    return {"cases": len(records), "answerable_cases": len(answerable),
            "unanswerable_cases": len(scored) - len(answerable), "k": k,
            "mean_recall": sum(row["recall"] for row in answerable) / len(answerable) if answerable else None,
            "hit_count": sum(row["hit"] for row in answerable),
            "mrr": sum(row["reciprocal_rank"] for row in answerable) / len(answerable) if answerable else None}
