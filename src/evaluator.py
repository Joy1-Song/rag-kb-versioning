def evaluate_keywords(answer: str, expected_keywords: list[str]) -> dict:
    if not expected_keywords:
        return {
            "hit_rate": 0.0,
            "hit_keywords": [],
            "missed_keywords": [],
        }

    hit_keywords = [keyword for keyword in expected_keywords if keyword in answer]
    missed_keywords = [
        keyword for keyword in expected_keywords if keyword not in hit_keywords
    ]

    return {
        "hit_rate": len(hit_keywords) / len(expected_keywords),
        "hit_keywords": hit_keywords,
        "missed_keywords": missed_keywords,
    }
