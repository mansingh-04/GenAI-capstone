"""Test script for Phase 4: Static RAG (ChromaDB + LIAR dataset)."""

import csv
import sys
import tempfile
from pathlib import Path

# Add milestone2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.static.liar_dataset_loader import LIARDatasetLoader
from rag.static.static_rag import StaticRAG


def create_sample_liar_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "1.json",
            "label": "false",
            "statement": "The government is hiding a secret technology.",
            "subject": "technology",
            "speaker": "Jane Doe",
            "job_title": "Researcher",
            "state": "California",
            "party": "independent",
            "barely_true_count": "0",
            "false_count": "1",
            "half_true_count": "0",
            "mostly_true_count": "0",
            "pants_fire_count": "0",
            "context": "a news article",
        },
        {
            "id": "2.json",
            "label": "true",
            "statement": "Scientists discovered a new frog species in the Amazon.",
            "subject": "science",
            "speaker": "Dr. Smith",
            "job_title": "Biologist",
            "state": "Washington",
            "party": "none",
            "barely_true_count": "0",
            "false_count": "0",
            "half_true_count": "0",
            "mostly_true_count": "1",
            "pants_fire_count": "0",
            "context": "scientific journal",
        },
        {
            "id": "3.json",
            "label": "pants-fire",
            "statement": "The stock market will rise by 50% this week.",
            "subject": "economy",
            "speaker": "John Analyst",
            "job_title": "Financial Advisor",
            "state": "New York",
            "party": "business",
            "barely_true_count": "0",
            "false_count": "0",
            "half_true_count": "0",
            "mostly_true_count": "0",
            "pants_fire_count": "1",
            "context": "social media",
        },
    ]

    with open(path, "w", newline="", encoding="utf-8") as tsvfile:
        fieldnames = ["id", "label", "statement", "subject", "speaker", "job_title",
                     "state", "party", "barely_true_count", "false_count", "half_true_count",
                     "mostly_true_count", "pants_fire_count", "context"]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        for row in rows:
            writer.writerow(row)


def test_liar_dataset_loader():
    print("\n" + "=" * 60)
    print("TEST 1: LIAR Dataset Loader")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "liar_dataset.tsv"
        create_sample_liar_tsv(temp_path)

        loader = LIARDatasetLoader(dataset_path=temp_path)
        records = loader.load_dataset()

        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        assert records[0]["claim"] == "The government is hiding a secret technology."
        assert records[1]["label"] == "true"

        print("✅ LIAR dataset loader works correctly")
        return True


def test_static_rag_index_and_query():
    print("\n" + "=" * 60)
    print("TEST 2: Static RAG Index and Query")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        sample_tsv = Path(temp_dir) / "liar_dataset.tsv"
        create_sample_liar_tsv(sample_tsv)

        chroma_dir = Path(temp_dir) / "chroma_db"
        rag = StaticRAG(
            chroma_db_path=chroma_dir,
            collection_name="test_fact_checks",
            top_k=2,
        )

        loader = LIARDatasetLoader(dataset_path=sample_tsv)
        records = loader.load_dataset()
        rag.index_fact_checks(records, reset=True)

        info = rag.get_collection_info()
        assert info["count"] == 3, f"Expected 3 indexed items, got {info['count']}"

        results = rag.query("government secret technology", top_k=2)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert any("secret technology" in result["claim"].lower() for result in results)

        print("✅ Static RAG indexing and query work correctly")
        return True


def run_all_tests():
    print("\n" + "🔬 " * 20)
    print("MILESTONE 2: PHASE 4 - STATIC RAG TEST SUITE")
    print("🔬 " * 20)

    results = {
        "LIAR Dataset Loader": test_liar_dataset_loader(),
        "Static RAG Index and Query": test_static_rag_index_and_query(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    if passed == total:
        print("\n🎉 All tests passed! Static RAG Phase 4 is complete.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
