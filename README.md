# PathNAVE

**PathNAVE** is a benchmark framework for evaluating **agentic pathology methods** on whole-slide image (WSI) analysis.

Unlike existing repositories that implement a single agent, **PathNAVE does not implement the agentic algorithms themselves**. Instead, it provides a unified evaluation framework that enables fair comparison across different agentic approaches under the same experimental setting.

## Supported Agentic Methods

This benchmark supports the following representative agentic pathology methods:

- **Pathology-CoT** (reimplemented from the original paper)
- **CPathAgent** (adapted from the official implementation)
- **PathAgent** (adapted from the official implementation)
- **SlideSeek** (reimplemented from the original paper)

Each method is converted into a unified trajectory representation consisting of:

- ROI navigation trajectory
- Evidence acquisition process
- Final diagnostic prediction

This unified representation allows all methods to be evaluated using the same metrics.

## Third-Party Implementations

Some methods included in this benchmark are based on publicly available implementations released by the original authors.

- `methods/CPathAgent/` — original README and license preserved.
- `methods/PathAgent/` — original README and license preserved.

Please refer to the corresponding directories for the original documentation and licensing information.

## Benchmark Components

The benchmark includes:

- Unified trajectory representation
- Navigation Quality evaluation
  - MRFH
  - ROI Hit Rate
  - ROI Precision
  - Navigation Cost
- Evidence Quality evaluation
- Final Answer Accuracy

## Repository Structure

```text
PathNAVE/
├── Agent_dataset/          # QA dataset included TCGA, CAMELONY16
├── Analyze/                # Result and plotting
├── CPathAgent/             # Method
├── PathAgent/              # Method
├── Pathology-CoT/          # Method
├── slide_seek/             # Method
├── benchmark_evaluation/   # Calculate the metric
└── README.md
```

## Acknowledgement

PathAgent and CPathAgent are based on the official implementations released by their original authors. We thank the authors for making their code publicly available.

PathNAVE builds a unified benchmarking pipeline on top of these methods to enable standardized evaluation and comparison.
