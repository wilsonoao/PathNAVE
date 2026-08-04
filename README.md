# PathNAVE

**PathNAVE** is a benchmark framework for evaluating **agentic pathology methods** on whole-slide image (WSI) analysis.

Unlike existing repositories that implement a single agent, **PathNAVE does not implement the agentic algorithms themselves**. Instead, it provides a unified evaluation framework that enables fair comparison across different agentic approaches under the same experimental setting.

## Supported Agentic Methods

This benchmark currently supports the evaluation of the following representative agentic pathology frameworks:

- **Pathology-CoT**
- **CPathAgent**
- **PathAgent**
- **SlideSeek**

Each method is converted into a unified trajectory representation consisting of:

- ROI navigation trajectory
- Evidence acquisition process
- Final diagnostic prediction

This unified representation allows all methods to be evaluated using the same metrics.

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
├── benchmark/          # Evaluation framework
├── datasets/           # Dataset interface
├── metrics/            # Navigation & Evidence metrics
├── visualization/      # Analysis & plotting
├── methods/            # Wrappers for supported agentic methods
└── README.md
```
