# PathNAVE

**PathNAVE** is a benchmark framework for systematically evaluating **agentic pathology methods** on whole-slide image (WSI) analysis.

Rather than proposing a new agentic method, PathNAVE focuses on **standardized evaluation and comparison** across representative agentic pathology frameworks. It provides a unified evaluation pipeline that enables fair and reproducible benchmarking under the same experimental setting.

---

## Supported Agentic Methods

This benchmark currently supports the following representative agentic pathology methods:

| Method | Implementation |
|--------|----------------|
| Pathology-CoT | Reimplemented from the original paper |
| CPathAgent | Adapted from the official implementation |
| PathAgent | Adapted from the official implementation |
| SlideSeek | Reimplemented from the original paper |

Each method is converted into a unified trajectory representation consisting of:

- ROI navigation trajectory
- Evidence acquisition process
- Final diagnostic prediction

This standardized representation enables all methods to be evaluated using the same navigation, evidence, and accuracy metrics.

---

## Third-Party Implementations

Some methods included in this benchmark are based on publicly available implementations released by their original authors.

The original README files and licenses are preserved in the corresponding directories:

- `CPathAgent/`
- `PathAgent/`

Please refer to the original documentation in these directories for implementation details and licensing information.

---

## Benchmark Components

PathNAVE provides:

- **Unified trajectory representation**
- **Navigation Quality Evaluation**
  - MRFH (Mean Reciprocal First Hit)
  - ROI Hit Rate
  - ROI Precision
  - Navigation Cost
- **Evidence Quality Evaluation**
- **Final Answer Accuracy**
- **Result analysis and visualization**

---

## Repository Structure

```text
PathNAVE/
├── Agent_dataset/          # QA datasets (TCGA, CAMELYON16)
├── Analyze/                # Result analysis and visualization
├── CPathAgent/             # Official implementation (adapted)
├── PathAgent/              # Official implementation (adapted)
├── Pathology-CoT/          # Reimplementation
├── slide_seek/             # Reimplementation
├── benchmark_evaluation/   # Benchmark metrics and evaluation
└── README.md
```

---

## Acknowledgement

PathAgent and CPathAgent are adapted from the official implementations released by their original authors. We thank the authors for making their code publicly available.

Pathology-CoT and SlideSeek are reimplemented based on the descriptions provided in their original publications.

PathNAVE builds a unified benchmarking pipeline to enable standardized and reproducible evaluation across different agentic pathology methods.
