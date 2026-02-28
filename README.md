# pdf_ocr

Convert PDFs to Markdown using vision-language models served by [vLLM](https://docs.vllm.ai/). Based on [batch-ocr-inference](https://github.com/fgbelidji/llm-lab/tree/main/batch-ocr-inference) by [@fgbelidji](https://github.com/fgbelidji).

Accepts a PDF file, a directory of PDFs, or a HuggingFace dataset as input. Outputs Markdown to local files, stdout, or a HuggingFace dataset repo.

## Usage

### CLI

```bash
# Single PDF to stdout
python -m pdf_ocr paper.pdf

# Directory of PDFs to local markdown files
python -m pdf_ocr ./pdfs/ --output ./output/

# Use an existing vLLM server
python -m pdf_ocr paper.pdf --base-url http://localhost:8000

# HuggingFace dataset input, push results to HF
python -m pdf_ocr hf://org/dataset --hf-repo org/output-dataset --private
```

### Python API

```python
from pdf_ocr import convert

results = convert("paper.pdf")
for result in results:
    for page in result.pages:
        print(page.markdown)

# With an existing vLLM server
results = convert("./pdfs/", base_url="http://localhost:8000", output="./output/")
```

## CLI Options

| Flag | Description |
|---|---|
| `source` | PDF file, directory, or HF dataset reference |
| `--model` | Model config name, HF model ID, or YAML config path (default: `lighton_ocr_2_1b`) |
| `--output`, `-o` | Local output directory |
| `--hf-repo` | HF repo ID to push results to |
| `--private` | Create private HF dataset (with `--hf-repo`) |
| `--base-url` | Existing vLLM server URL (skips server launch) |
| `--batch-size` | Override inference batch size |
| `--max-pages` | Limit total pages processed |
| `--pdf-column` | Column name for PDFs in HF datasets |
| `--split` | HF dataset split (default: `train`) |
| `--port` | vLLM server port (default: `8000`) |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

## Model Configuration

Models are configured via YAML files in `models/`. The bundled config (`lighton_ocr_2_1b`) uses [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B). Custom configs can be passed by path:

```bash
python -m pdf_ocr paper.pdf --model ./my_model.yaml
```

Config structure:

```yaml
model_id: lightonai/LightOnOCR-2-1B
served_model_name: lighton-ocr

vllm_args:
  max-model-len: 4096
  gpu-memory-utilization: 0.90
  trust-remote-code: true

inference:
  max_tokens: 4096
  temperature: 0.2
  batch_size: 4

pdf_rendering:
  dpi: 200
  max_dimension: 1540
```

## Package Structure

```
pdf_ocr/
├── __init__.py      # Public convert() API
├── __main__.py      # python -m entry point
├── cli.py           # Argument parsing
├── config.py        # YAML model config loading
├── convert.py       # Batch conversion pipeline
├── pdf_input.py     # PDF loading/rendering (files, dirs, HF datasets)
├── server.py        # vLLM server lifecycle and inference client
├── storage.py       # Output to local files or HF Hub
└── models/
    └── lighton_ocr_2_1b.yaml
```
