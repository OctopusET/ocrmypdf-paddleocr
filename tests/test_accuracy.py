# SPDX-License-Identifier: MPL-2.0

"""OCR accuracy tests using real documents with ground truth."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RESOURCES = Path(__file__).parent / "resources"


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate using edit distance."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Simple Levenshtein distance
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[m] / n


def word_accuracy(reference: str, hypothesis: str) -> float:
    """Fraction of reference words found in hypothesis."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words:
        return 1.0
    return len(ref_words & hyp_words) / len(ref_words)


class TestLinnAccuracy:
    """Accuracy test using linn.pdf -- a scanned Linn hardware brochure."""

    @pytest.fixture(scope="class")
    def ground_truth(self):
        return (RESOURCES / "linn.txt").read_text()

    @pytest.fixture(scope="class")
    def ocr_text(self, tmp_path_factory):
        """Run OCRmyPDF pipeline and extract text from output."""
        import ocrmypdf

        tmp = tmp_path_factory.mktemp("linn")
        output = tmp / "linn_ocr.pdf"

        result = ocrmypdf.ocr(
            str(RESOURCES / "linn.pdf"),
            str(output),
            plugins=["ocrmypdf_paddleocr"],
            language=["eng"],
        )
        assert result == 0

        extracted = subprocess.run(
            ["pdftotext", str(output), "-"],
            capture_output=True,
            text=True,
        )
        return extracted.stdout

    def test_word_accuracy_above_threshold(self, ground_truth, ocr_text):
        """At least 50% of ground truth words should be found in OCR output."""
        acc = word_accuracy(ground_truth, ocr_text)
        assert acc >= 0.50, f"Word accuracy {acc:.1%} below 50% threshold"

    def test_key_phrases_detected(self, ocr_text):
        """Key phrases from the document should appear in OCR output."""
        text = ocr_text.lower()
        assert "linnsequencer" in text.replace(" ", "").replace("-", "")
        assert "midi" in text
        assert "record" in text

    def test_cer_reasonable(self, ground_truth, ocr_text):
        """Character Error Rate should be under 50%."""
        rate = cer(ground_truth, ocr_text)
        assert rate < 0.50, f"CER {rate:.1%} above 50% threshold"


class TestKoreanOcr:
    """Basic Korean OCR test using PaddleOCR sample image."""

    def test_korean_text_detected(self):
        """PaddleOCR should detect Korean text from its own sample image."""
        from unittest.mock import MagicMock

        from ocrmypdf_paddleocr.engine import PaddleOcrEngine

        img = RESOURCES / "korean_1.jpg"
        if not img.exists():
            pytest.skip("korean_1.jpg not available")

        engine = PaddleOcrEngine()
        options = MagicMock()
        options.languages = ["kor"]

        ocr_tree, text = engine.generate_ocr(img, options)
        assert len(text) > 0, "No text detected in Korean image"
        assert len(ocr_tree.children) > 0, "No OCR elements found"
