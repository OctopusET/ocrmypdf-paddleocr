# SPDX-License-Identifier: MPL-2.0

"""Tests for PaddleOCR engine plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ocrmypdf_paddleocr import check_options, get_ocr_engine
from ocrmypdf_paddleocr.engine import PaddleOcrEngine
from ocrmypdf_paddleocr.lang_map import SUPPORTED_LANGUAGES, tesseract_to_paddle


class TestLangMap:
    def test_common_languages(self):
        assert tesseract_to_paddle('eng') == 'en'
        assert tesseract_to_paddle('kor') == 'korean'
        assert tesseract_to_paddle('chi_sim') == 'ch'
        assert tesseract_to_paddle('jpn') == 'japan'
        assert tesseract_to_paddle('fra') == 'french'

    def test_unknown_passthrough(self):
        assert tesseract_to_paddle('xyz') == 'xyz'

    def test_supported_languages_nonempty(self):
        assert len(SUPPORTED_LANGUAGES) > 20


class TestPaddleOcrEngine:
    @pytest.fixture
    def engine(self):
        return PaddleOcrEngine()

    def test_version(self):
        v = PaddleOcrEngine.version()
        assert isinstance(v, str)
        assert v != ''

    def test_creator_tag(self, engine):
        options = MagicMock()
        tag = engine.creator_tag(options)
        assert 'PaddleOCR' in tag

    def test_str(self, engine):
        s = str(engine)
        assert 'PaddleOCR' in s

    def test_languages(self, engine):
        options = MagicMock()
        langs = engine.languages(options)
        assert 'eng' in langs
        assert 'kor' in langs

    def test_orientation(self, engine, tmp_path):
        img_path = tmp_path / 'upright.png'
        img = Image.new('RGB', (200, 200), color='white')
        img.save(img_path)
        options = MagicMock()
        result = engine.get_orientation(img_path, options)
        assert result.angle in (0, 90, 180, 270)
        assert 0.0 <= result.confidence <= 15.0

    def test_deskew(self, engine, tmp_path):
        img_path = tmp_path / 'straight.png'
        img = Image.new('RGB', (612, 792), color='white')
        img.save(img_path, dpi=(300, 300))
        options = MagicMock()
        options.languages = ['eng']
        options.paddleocr_use_gpu = False
        angle = engine.get_deskew(img_path, options)
        assert isinstance(angle, float)

    def test_supports_generate_ocr(self):
        assert PaddleOcrEngine.supports_generate_ocr() is True

    def test_generate_ocr_on_image(self, engine, tmp_path):
        """Test OCR on a simple white image (should return empty or minimal text)."""
        img_path = tmp_path / 'blank.png'
        img = Image.new('RGB', (612, 792), color='white')
        img.save(img_path, dpi=(300, 300))

        options = MagicMock()
        options.languages = ['eng']
        options.paddleocr_use_gpu = False

        ocr_tree, text = engine.generate_ocr(img_path, options, page_number=0)

        assert ocr_tree.ocr_class == 'ocr_page'
        assert ocr_tree.bbox is not None
        assert ocr_tree.bbox.right == 612
        assert ocr_tree.bbox.bottom == 792
        assert abs(ocr_tree.dpi - 300.0) < 1.0
        assert isinstance(text, str)

    def test_generate_hocr_raises(self, engine):
        with pytest.raises(NotImplementedError):
            engine.generate_hocr(None, None, None, None)

    def test_generate_pdf_raises(self, engine):
        with pytest.raises(NotImplementedError):
            engine.generate_pdf(None, None, None, None)

    def test_check_options_forces_jobs_1(self):
        """PaddlePaddle isn't multi-process safe, so we force jobs=1."""
        options = MagicMock()
        options.jobs = 8
        check_options(options)
        assert options.jobs == 1

    def test_get_ocr_engine_returns_engine(self):
        engine = get_ocr_engine()
        assert isinstance(engine, PaddleOcrEngine)
