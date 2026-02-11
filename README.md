# Parameter-Efficient CQNN: A Hybrid Knowledge Distillation Approach from Classical CNN to QNN

## Overview

This project addresses the fundamental challenge of building effective Quantum Neural Networks (QNNs) under the constraints of Noisy Intermediate-Scale Quantum (NISQ) devices. Current quantum hardware is severely limited by small qubit counts, high error rates, and restricted quantum coherence times, making it difficult to scale quantum models to practical performance levels. Direct training of low-qubit QNNs typically results in poor accuracy due to limited state-space dimensionality and unstable gradient dynamics.

Our work introduces a cross-domain knowledge distillation framework that enables compact quantum models to achieve substantially higher accuracy without requiring additional quantum resources. By transferring structured representations from a classical teacher network to a quantum student network, we demonstrate that training methodology can be more decisive than circuit size in near-term quantum machine learning applications.

## Problem Statement

Quantum Machine Learning (QML) leverages quantum mechanical properties such as superposition and entanglement to process information in exponentially large Hilbert spaces, offering theoretical advantages over classical approaches for pattern recognition, optimization, and high-dimensional data analysis. However, practical implementation faces critical bottlenecks:

- **Limited Qubit Availability**: Current quantum hardware typically provides fewer than 100 qubits, restricting model capacity
- **High Error Rates**: NISQ devices exhibit significant noise and decoherence, degrading quantum state fidelity
- **Training Instability**: Low-qubit Parameterized Quantum Circuits (PQCs) suffer from volatile gradients and poor convergence
- **Exponential Scaling Cost**: Each additional qubit doubles Hilbert space dimensionality, creating exponential simulation overhead

These constraints necessitate parameter-efficient approaches that maximize the learning capability of resource-constrained quantum models.

## Methodology


### Architecture Design

Our framework implements a hybrid classical-quantum knowledge distillation pipeline consisting of three core components:

**1. Teacher Network (Classical CNN)**
- Architecture: Modified LeNet-5 convolutional neural network
- Modification: Adapted for 28×28 MNIST inputs by removing the third convolutional layer
- Role: Generates soft targets (probabilistic logits) that encode rich supervisory signals

**2. Student Network (4-Qubit Hybrid QNN)**
- Classical Component: Convolutional feature extraction layer
- Quantum Component: 4-qubit Parameterized Quantum Circuit (PQC)
- Hilbert Space: 2⁴ = 16 dimensional quantum state space
- Role: Learns from both ground truth labels and teacher's soft predictions

**3. Distillation Loss Function**
- Combines standard cross-entropy loss with KL-divergence between student and teacher logits
- Temperature parameter controls softness of probability distributions
- Enables gradient flow from teacher's confident predictions to guide student optimization

### Training Procedure

The methodology follows a two-phase training protocol:

**Phase 1: Teacher Training**
- Train LeNet-5 on full 10-class MNIST dataset using standard supervised learning
- Achieve high accuracy (98.38%) to establish reliable soft targets
- Generate informative logit distributions for knowledge transfer

**Phase 2: Student Distillation**
- Initialize 4-qubit hybrid QNN with random parameters
- Train using combined loss: distillation loss (teacher guidance) + classification loss (ground truth)
- Leverage teacher's smoothed probability distributions to stabilize quantum gradient updates
- Parameter updates via parameter-shift rule for differentiable PQC training

### Data Processing

- **Dataset**: MNIST handwritten digits (60,000 training, 10,000 testing images)
- **Preprocessing**: Normalization, 8×8 patch extraction for quantum encoding
- **Quantum Encoding**: Classical features mapped to quantum states via amplitude encoding
- **Class Distribution**: Uniform across 10 classes (no augmentation required)

## Key Results

The experimental evaluation demonstrates the efficacy of knowledge distillation for parameter-efficient quantum learning:

| Model Configuration | Accuracy | Improvement | Qubit Count | Hilbert Dimension |
|---------------------|----------|-------------|-------------|-------------------|
| 4-Qubit QNN (Baseline) | 65.74% | — | 4 | 16 |
| 4-Qubit QNN (Distilled) | **86.10%** | **+20.36%** | 4 | 16 |
| 8-Qubit QNN (Baseline) | 96.38% | +10.28% | 8 | 256 |
| Teacher (LeNet-5) | 98.38% | — | 0 | N/A |

### Critical Findings

1. **Distillation Impact**: Knowledge distillation improved 4-qubit QNN accuracy by 20.36 percentage points without increasing quantum resource requirements

2. **Resource Efficiency**: The distilled 4-qubit model achieved 86.10% accuracy, approaching the 8-qubit baseline (96.38%) while using 16× smaller Hilbert space, demonstrating superior accuracy-cost efficiency

3. **Training Stability**: Incorporating teacher soft targets substantially stabilized convergence dynamics, mitigating the optimization volatility characteristic of low-qubit PQCs

4. **Practical Viability**: The 4-qubit distilled configuration delivers optimal performance for near-term quantum hardware where qubit availability and simulation costs remain prohibitive constraints

## Technical Implementation

### Platform Requirements

- **Computational Environment**: Google Colab Pro with GPU acceleration
- **Hardware**: NVIDIA A100-SXM4 Tensor Core GPU (80 GB HBM2e)
- **Software Stack**:
  - PyTorch: Classical model development and training orchestration
  - TorchQuantum: GPU-accelerated state-vector simulation of PQCs
  - IBM Qiskit: Quantum circuit visualization and ansatz verification
  - TorchVision: MNIST data acquisition and preprocessing

### Workflow Architecture

The implementation adopts a hybrid classical-quantum design where PyTorch handles data preprocessing and teacher training, while TorchQuantum enables differentiable PQC training through tensorized gate operations and parameter-shift gradient computation.

## Conclusion

This work establishes that training signal quality—achieved through knowledge distillation—can be more decisive than model size in resource-constrained quantum machine learning settings. The results demonstrate a viable pathway for deploying effective quantum models on near-term NISQ devices by leveraging classical teacher networks to enhance quantum student performance without exponential resource scaling. The distilled 4-qubit QNN represents the most practical candidate for current quantum hardware, achieving high accuracy while maintaining minimal qubit overhead.

---

**Course**: CIS 660 - Quantum Machine Learning  
**Author**: Md Shawmoon Azad  
**Affiliation**: PhD Student in Computer Science, Cleveland State University
