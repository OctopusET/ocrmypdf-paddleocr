# OCRmyPDF PaddleOCR

A plugin to use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) as the
OCR engine for [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF), instead of
Tesseract. PaddleOCR generally provides better accuracy, especially for CJK
languages (Chinese, Japanese, Korean).

Unlike the EasyOCR plugin, this plugin does not require Tesseract for any
operations.

## Installation

```bash
pip install ocrmypdf-paddleocr
```

## Usage

```bash
ocrmypdf --plugin ocrmypdf_paddleocr input.pdf output.pdf
ocrmypdf --plugin ocrmypdf_paddleocr -l kor input.pdf output.pdf
```

Or from Python:

```python
import ocrmypdf

ocrmypdf.ocr('input.pdf', 'output.pdf', plugins=['ocrmypdf_paddleocr'])
```

## Known limitations

- The plugin forces `jobs=1`. PaddlePaddle's inference engine already uses
  all CPU cores internally, so parallel page processing would only cause
  contention

## License

[MPL-2.0](LICENSE)
