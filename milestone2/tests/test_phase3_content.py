"""
Test script for Phase 3: Content Extraction

Run this to verify content extraction functionality works correctly.
"""

import sys
from pathlib import Path

# Add milestone2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.content_extraction_tools import (
    extract_content,
    clean_article_text,
    validate_url,
    batch_extract_content,
)
from logger import logger_tools


def test_url_validation():
    """Test 1: URL Validation"""
    print("\n" + "=" * 60)
    print("TEST 1: URL Validation")
    print("=" * 60)

    test_cases = [
        ("https://www.bbc.com/news", True),
        ("http://example.com/article", True),
        ("not-a-url", False),
        ("ftp://invalid.com", False),
        ("", False),
        ("https://", False),
    ]

    passed = 0
    for url, expected in test_cases:
        result = validate_url(url)
        status = "✅" if result["is_valid"] == expected else "❌"
        print(f"{status} {url[:40]:40} → {result['is_valid']}")
        if result["is_valid"] == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_text_cleaning():
    """Test 2: Text Cleaning"""
    print("\n" + "=" * 60)
    print("TEST 2: Text Cleaning")
    print("=" * 60)

    test_cases = [
        (
            "Check   out    this!!!   text",
            "Check out this",
            "Multiple spaces and special chars removed",
        ),
        (
            "Visit http://example.com for more",
            "Visit for more",
            "URLs removed",
        ),
        (
            "Test@#$%^&*(text",
            "Test(text",
            "Dangerous special chars removed, safe punctuation kept",
        ),
    ]

    passed = 0
    for input_text, expected_substring, description in test_cases:
        result = clean_article_text(input_text)
        success = expected_substring in result
        status = "✅" if success else "❌"
        print(f"{status} {description}: '{expected_substring}' in '{result}'")
        if success:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_raw_text_extraction():
    """Test 3: Raw Text Extraction"""
    print("\n" + "=" * 60)
    print("TEST 3: Raw Text Extraction")
    print("=" * 60)

    # Sample news articles
    real_news = """
    Scientists Discover New Species in Amazon Rainforest
    
    A team of researchers from the University of São Paulo announced today the 
    discovery of a previously unknown species of frog in the Amazon rainforest.
    The species was identified during a biodiversity survey conducted over the 
    past two years. According to the research, the frog exhibits unique 
    characteristics that distinguish it from other known species in the region.
    
    The discovery adds to the growing list of new species found in the Amazon,
    highlighting the importance of continued conservation efforts.
    """

    fake_news = """
    SHOCKING REVELATION: Government Hiding Secret Technology!
    
    In a shocking turn of events, sources claim the government has been hiding
    an advanced technology from the public for decades! This is the biggest
    scandal of our time! Everyone must know about this immediately! Share this
    everywhere! The truth is finally coming out!
    """

    tests = [
        (real_news, "real news"),
        (fake_news, "sensational news"),
    ]

    passed = 0
    for text, label in tests:
        try:
            result = extract_content(text, input_type="text")

            if result["success"]:
                print(f"✅ {label} extracted successfully")
                print(f"   Source: {result['metadata']['source']}")
                print(f"   Length: {result['metadata']['length']} chars")
                print(f"   Words: {result['metadata']['word_count']}")
                passed += 1
            else:
                print(f"❌ {label} extraction failed: {result['error']}")

        except Exception as e:
            print(f"❌ {label} test failed: {str(e)}")

    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_input_validation():
    """Test 4: Input Validation"""
    print("\n" + "=" * 60)
    print("TEST 4: Input Validation")
    print("=" * 60)

    # Invalid inputs
    invalid_inputs = [
        ("", "Empty string"),
        ("  ", "Only spaces"),
        ("Short", "Too short text"),
    ]

    passed = 0
    for input_data, description in invalid_inputs:
        try:
            result = extract_content(input_data)
            if not result["success"]:
                print(f"✅ {description} correctly rejected")
                passed += 1
            else:
                print(f"❌ {description} should have been rejected")
        except Exception as e:
            print(f"✅ {description} correctly rejected (exception)")
            passed += 1

    print(f"\nPassed: {passed}/{len(invalid_inputs)}")
    return passed == len(invalid_inputs)


def test_batch_extraction():
    """Test 5: Batch Extraction"""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Extraction")
    print("=" * 60)

    batch_inputs = [
        {
            "data": """
            Article One: Breaking news about technology.
            Scientists have discovered something amazing today.
            This is a longer text to meet minimum length requirement.
            This article contains important information for everyone.
            Please read this carefully and share with others.
            """
        },
        {
            "data": """
            Article Two: More breaking news.
            Something happened today that everyone should know about.
            This is another test article to verify batch processing works.
            Batch processing is important for handling multiple items.
            Let's verify this works correctly in all cases.
            """
        },
        {
            "data": "Short text",  # This should fail
        },
    ]

    try:
        results = batch_extract_content(batch_inputs)

        successful = sum(1 for r in results if r["success"])
        print(f"✅ Batch extraction completed")
        print(f"   Total: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")

        return successful >= 2  # At least 2 should succeed

    except Exception as e:
        print(f"❌ Batch extraction failed: {str(e)}")
        return False


def test_auto_detection():
    """Test 6: Auto-detection of input type"""
    print("\n" + "=" * 60)
    print("TEST 6: Auto-detection of Input Type")
    print("=" * 60)

    # Text (should be auto-detected as text)
    text = """
    This is a test article that should be auto-detected as text.
    It's longer than minimum length and contains actual content.
    Auto-detection should identify this as raw text input correctly.
    This needs to be long enough to pass minimum length requirements.
    """

    try:
        result = extract_content(text, input_type="auto")

        if result["success"] and result["metadata"]["source"] == "raw_text":
            print("✅ Text auto-detected correctly")
            print(f"   Detected as: {result['metadata']['source']}")
            return True
        else:
            print("❌ Text auto-detection failed")
            return False

    except Exception as e:
        print(f"❌ Auto-detection test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🔬 " * 20)
    print("MILESTONE 2: PHASE 3 - CONTENT EXTRACTION TEST SUITE")
    print("🔬 " * 20)

    results = {
        "URL Validation": test_url_validation(),
        "Text Cleaning": test_text_cleaning(),
        "Raw Text Extraction": test_raw_text_extraction(),
        "Input Validation": test_input_validation(),
        "Batch Extraction": test_batch_extraction(),
        "Auto-detection": test_auto_detection(),
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
        print("\n🎉 All tests passed! Phase 3 Content Extraction is complete.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
