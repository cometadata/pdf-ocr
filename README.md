# pdf-ocr

Convert PDFs to Markdown using vision-language models via [vLLM](https://docs.vllm.ai/). Based on [batch-ocr-inference](https://github.com/fgbelidji/llm-lab/tree/main/batch-ocr-inference) by [@fgbelidji](https://github.com/fgbelidji).

Accepts a PDF file, a directory of PDFs, or a HuggingFace dataset as input. Outputs Markdown to local files, stdout, or a HuggingFace dataset repo. Supports server and offline inference backends with automatic multi-GPU data parallelism.

## Usage

### CLI

```bash
# Single PDF to stdout
pdf_ocr paper.pdf

# Directory of PDFs to local markdown files
pdf_ocr ./pdfs/ --output ./output/

# Offline mode (in-process vLLM, auto multi-GPU)
pdf_ocr paper.pdf --backend offline --output ./output/

# Use an existing vLLM server
pdf_ocr paper.pdf --base-url http://localhost:8000

# HuggingFace dataset input, push results to HF
pdf_ocr hf://org/dataset --hf-repo org/output-dataset --private
```

### Python API

```python
from pdf_ocr import convert

results = convert("paper.pdf")
for result in results:
    for page in result.pages:
        print(page.markdown)

# Offline mode with an output directory
results = convert("./pdfs/", backend="offline", output="./output/")

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
| `--backend` | `server` (default) or `offline` (in-process vLLM) |
| `--batch-size` | Override inference batch size |
| `--max-pages` | Limit total pages processed |
| `--pdf-column` | Column name for PDFs in HF datasets |
| `--split` | HF dataset split (default: `train`) |
| `--port` | vLLM server port (default: `8000`) |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

## Inference Backends

**Server mode** (default): Launches a vLLM server and sends requests via the OpenAI-compatible API. Automatically detects multiple GPUs and sets `--data-parallel-size` to match. The client uses async requests with a configurable concurrency semaphore and exponential-backoff retries. Use `--base-url` to connect to an existing server instead.

**Offline mode** (`--backend offline`): Runs vLLM directly in-process. Automatically detects multiple GPUs and distributes batches across them via round-robin data parallelism with automatic dead-worker detection and restart. GPU-specific vLLM settings (memory utilization, batched tokens) are auto-tuned based on detected VRAM. Install with `pip install .[offline]` to include the vLLM dependency.

## Model Configuration

Models are configured via YAML files in `models/`. The bundled config (`lighton_ocr_2_1b`) uses [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B). To use a different model, create a YAML config and pass it by path:

```yaml
# my_model.yaml — only model_id is required; everything else has defaults
model_id: your-org/your-model
```

```bash
pdf_ocr paper.pdf --model ./my_model.yaml
pdf_ocr ./pdfs/ --model ./my_model.yaml --backend offline --output ./output/
```

Full config reference (all fields shown with their defaults):

```yaml
model_id: lightonai/LightOnOCR-2-1B
served_model_name: lighton-ocr

vllm_args:
  limit-mm-per-prompt: '{"image": 1}'
  no-enable-prefix-caching: true
  mm-processor-cache-gb: 0
  enable-chunked-prefill: true
  max-model-len: 4096
  gpu-memory-utilization: 0.90
  trust-remote-code: true

inference:
  max_tokens: 4000
  temperature: 0.2
  top_p: 0.9
  batch_size: 4              # server mode batch size
  max_concurrency: 4         # concurrent server requests
  request_timeout: 120
  max_retries: 3
  retry_backoff: 2.0
  offline_batch_size: 128    # offline mode batch size
  render_workers: 4          # parallel PDF rendering threads
  max_retry_depth: 3         # recursive subdivision on failure
  flush_every: 10            # batches between HF Hub uploads

pdf_rendering:
  dpi: 200
  max_dimension: 1540
```

## Package Structure

```
pdf_ocr/
├── __init__.py          # Public convert() API
├── __main__.py          # python -m entry point
├── cli.py               # Argument parsing
├── config.py            # YAML model config loading
├── convert.py           # Streaming batch conversion with checkpoint/resume
├── engine_factory.py    # Creates offline inference engine
├── gpu.py               # GPU detection and VRAM-based auto-tuning
├── offline.py           # In-process vLLM engine with multi-GPU data parallelism
├── pdf_input.py         # PDF loading/rendering (files, dirs, HF datasets)
├── server.py            # vLLM server lifecycle and async OpenAI-compatible client
├── storage.py           # Output to local files or HF Hub, checkpointing
├── models/
│   └── lighton_ocr_2_1b.yaml
└── hf_jobs/
    └── hf_job_runner.py # Entry point for HuggingFace Jobs execution
```
