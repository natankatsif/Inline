# Inline: Logic-Preserving Context Compression

**Inline** is a hybrid context compression framework for LLMs that maximizes prompt window utility. Unlike statistical methods, Inline prioritizes **syntactic integrity**, preventing "semantic collapse" and hallucinations in large data.

> "Fewer tokens result in stronger AI performance by minimizing noise and maximizing focal logic."

---

## 🚀 Key Features

* **Syntactic Tiering:** Deterministic analysis (spaCy) separates text into "VIP-tokens" (subjects, negations, roots) and "linguistic water".
* **Adaptive Stop-Loss:** Automatic compression halt upon a sharp spike in perplexity ($\Delta PP > 1.5$), ensuring readability.
* **Structural Guardrails:** Safe processing of JSON, Python, and SQL via AST validation — code remains functional after compression.
* **Cascaded Architecture:** Utilization of lightweight models (SLMs) as pre-processors for heavy LLMs.

---

## 📊 Results (Inline vs Selective Context)

Based on stress tests across Logic, Legal, and Structural domains:

| Metric | Selective Context | **Inline (Ours)** | Improvement |
| --- | --- | --- | --- |
| **Compression Ratio** | 5.0% | **25.2%** | **+404%** |
| **Avg. Latency** | 2.68s | **0.09s** | **~30x Faster** |
| **Failure Rate** | 20.0% | **0.0%** | **Absolute Stability** |
| **Logic Score** | 0.50 | **0.94** | **+88%** |

---

## 🛠 How it Works

The algorithm calculates the **Functional Importance Score** ($M$):

$$M(x_t) = \omega(dep_t, pos_t) \cdot I(x_t)$$

Where:

* $\omega$ — weight based on syntactic role (e.g., $\omega \approx 10.0$ for negations).
* $I(x_t)$ — informational surprisal of the token.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/inline.git
cd inline
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## 💻 Quick Start

```python
from inline import Inline

compressor = Inline(spacy_model="en_core_web_md", gpt2_model="gpt2")

raw_context = """
Although the user has explicitly requested a refund, the system 
must NOT process it unless the transaction ID is verified manually.
"""

compressed_text = compressor.compress(raw_context)

print(f"Original: {len(raw_context)} chars")
print(f"Compressed: {len(compressed_text)} chars")
print(f"Output: {compressed_text}")
```

---

## 📂 Repository Structure

* `inline.py` — Core source code (Syntactic parser, Observer).
* `/assets` — Charts and performance visualization.
* `paper.md` — Full text of the research paper.

---

## 📝 Citation

If you use Inline in your research, please cite:

```bibtex
@article{katsif2026inline,
  title={Inline: Logic-Preserving Context Compression via Syntactic Tiering},
  author={Katsif, Nataneli},
  journal={Orizont Durlesti Research},
  year={2026}
}
```
