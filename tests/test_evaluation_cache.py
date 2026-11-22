import pytest

from marsrag.evaluation.cache import (
    CacheEvalDataset,
    cosine_similarity,
    evaluate_cache_thresholds,
)


def _dataset():
    return CacheEvalDataset(
        dataset_name="cache-test",
        pairs=[
            {"id": "same", "cached_question": "What is the crew size?",
             "candidate_question": "How many crew members?", "equivalent": True},
            {"id": "different", "cached_question": "What is the crew size?",
             "candidate_question": "What is the radiation limit?", "equivalent": False},
        ],
    )


def test_evaluate_cache_thresholds_reports_false_positives_and_recall():
    scores = {("What is the crew size?", "How many crew members?"): 0.94,
              ("What is the crew size?", "What is the radiation limit?"): 0.91}
    report = evaluate_cache_thresholds(
        _dataset(), lambda left, right: scores[(left, right)],
        thresholds=(0.90, 0.93, 0.95, 0.97),
    )

    assert report["thresholds"]["0.9"]["counts"]["true_positive"] == 1
    assert report["thresholds"]["0.9"]["counts"]["false_positive"] == 1
    assert report["thresholds"]["0.93"]["metrics"]["precision"] == 1.0
    assert report["thresholds"]["0.95"]["metrics"]["recall"] == 0.0
    assert report["thresholds"]["0.97"]["metrics"]["false_positive_rate"] == 0.0


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize("left,right", [([], []), ([1.0], [1.0, 2.0]), ([0.0], [1.0])])
def test_cosine_similarity_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def test_cache_dataset_requires_both_pair_classes():
    with pytest.raises(ValueError):
        CacheEvalDataset(
            dataset_name="invalid",
            pairs=[{"id": "only", "cached_question": "Question one?",
                    "candidate_question": "Question two?", "equivalent": True}],
        )


def test_thresholds_must_be_valid():
    with pytest.raises(ValueError):
        evaluate_cache_thresholds(_dataset(), lambda _a, _b: 0.9, thresholds=(0.0,))

def test_load_cache_dataset_accepts_utf8_bom(tmp_path):
    from marsrag.evaluation.cache import load_cache_dataset

    payload = {
        "schema_version": 1,
        "dataset_name": "bom-test",
        "pairs": [
            {
                "id": "same",
                "cached_question": "What is the crew size?",
                "candidate_question": "How many crew members?",
                "equivalent": True,
            },
            {
                "id": "different",
                "cached_question": "What is the crew size?",
                "candidate_question": "What is the radiation limit?",
                "equivalent": False,
            },
        ],
    }
    path = tmp_path / "cache_dataset.json"
    path.write_bytes(b"\xef\xbb\xbf" + __import__("json").dumps(payload).encode("utf-8"))

    dataset = load_cache_dataset(path)

    assert dataset.dataset_name == "bom-test"
    assert len(dataset.pairs) == 2


def test_load_cache_dataset_accepts_utf8_without_bom(tmp_path):
    import json

    from marsrag.evaluation.cache import load_cache_dataset

    payload = {
        "dataset_name": "utf8-test",
        "pairs": [
            {
                "id": "same",
                "cached_question": "What is the crew size?",
                "candidate_question": "How many crew members?",
                "equivalent": True,
            },
            {
                "id": "different",
                "cached_question": "What is the crew size?",
                "candidate_question": "What is the radiation limit?",
                "equivalent": False,
            },
        ],
    }
    path = tmp_path / "cache_dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_cache_dataset(path)

    assert dataset.dataset_name == "utf8-test"
    assert len(dataset.pairs) == 2

