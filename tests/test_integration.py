# SPDX-License-Identifier: MPL-2.0

"""Integration tests for the PaddleOCR OCRmyPDF plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from ocrmypdf_paddleocr.engine import PaddleOcrEngine


@pytest.fixture(scope="module")
def text_image(tmp_path_factory):
    """Create a test image with known text."""
    img_path = tmp_path_factory.mktemp("images") / "text.png"
    img = Image.new("RGB", (612, 792), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 50), "Hello World", fill="black")
    d.text((50, 100), "This is a test document.", fill="black")
    d.text((50, 150), "1234567890", fill="black")
    img.save(img_path, dpi=(300, 300))
    return img_path


@pytest.fixture(scope="module")
def engine():
    return PaddleOcrEngine()


@pytest.fixture(scope="module")
def options():
    opts = MagicMock()
    opts.languages = ["eng"]
    opts.paddleocr_use_gpu = False
    return opts


class TestOcrAccuracy:
    def test_detects_text(self, engine, text_image, options):
        """Engine should detect text in the image."""
        ocr_tree, text = engine.generate_ocr(text_image, options)
        assert len(text) > 0
        assert len(ocr_tree.children) > 0

    def test_text_content(self, engine, text_image, options):
        """Recognized text should contain expected words."""
        _, text = engine.generate_ocr(text_image, options)
        assert "Hello" in text
        assert "1234567890" in text

    def test_word_bboxes_valid(self, engine, text_image, options):
        """All word bounding boxes should be within page bounds."""
        ocr_tree, _ = engine.generate_ocr(text_image, options)
        page_w = ocr_tree.bbox.right
        page_h = ocr_tree.bbox.bottom

        for line in ocr_tree.children:
            for word in line.children:
                assert word.bbox.left >= 0
                assert word.bbox.top >= 0
                assert word.bbox.right <= page_w
                assert word.bbox.bottom <= page_h
                assert word.bbox.right > word.bbox.left
                assert word.bbox.bottom > word.bbox.top

    def test_confidence_range(self, engine, text_image, options):
        """Confidence scores should be between 0 and 1."""
        ocr_tree, _ = engine.generate_ocr(text_image, options)
        for line in ocr_tree.children:
            for word in line.children:
                assert 0.0 <= word.confidence <= 1.0


class TestFullPipeline:
    def test_ocrmypdf_pipeline(self, tmp_path, text_image):
        """Test full OCRmyPDF pipeline: PDF metadata, text extraction, multi-page."""
        import subprocess

        import img2pdf

        import ocrmypdf
        import pikepdf

        # Create single-page PDF
        sample_pdf = tmp_path / "input.pdf"
        sample_pdf.write_bytes(img2pdf.convert(str(text_image)))

        output = tmp_path / "output.pdf"
        result = ocrmypdf.ocr(
            str(sample_pdf),
            str(output),
            plugins=["ocrmypdf_paddleocr"],
            language=["eng"],
        )
        assert result == 0
        assert output.exists()
        assert output.stat().st_size > 0

        # Check PDF metadata
        with pikepdf.open(output) as pdf:
            assert len(pdf.pages) == 1
            assert pdf.pages[0].Contents is not None
            creator = str(pdf.docinfo.get("/Creator", ""))
            assert "PaddleOCR" in creator

        # Round-trip text extraction
        extracted = subprocess.run(
            ["pdftotext", str(output), "-"],
            capture_output=True,
            text=True,
        )
        text = extracted.stdout
        assert "Hello" in text
        assert "1234567890" in text

        # Multi-page PDF (reuses engine singleton within same pipeline)
        multipage_pdf = tmp_path / "multipage.pdf"
        multipage_pdf.write_bytes(
            img2pdf.convert([str(text_image), str(text_image)])
        )
        output_multi = tmp_path / "output_multi.pdf"
        result = ocrmypdf.ocr(
            str(multipage_pdf),
            str(output_multi),
            plugins=["ocrmypdf_paddleocr"],
            language=["eng"],
        )
        assert result == 0
        with pikepdf.open(output_multi) as pdf:
            assert len(pdf.pages) == 2
