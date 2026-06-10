# Beyond Scale: A Revision Theory of Intelligence (RTI)

Official implementation and mathematical framework for the **Revision Theory of Intelligence (RTI)**, a new paradigm designed to achieve zero-shot out-of-distribution (OOD) adaptation via dynamic structural overrides rather than parametric gradient descent.

## 🚀 Key Framework Pillars

* **Contradiction-Driven Activation:** Bypasses passive autoregressive next-token prediction by wrapping the world model core in a dynamic statistical confidence filter.
* **Dual-Route Architecture:** Splits memory into a static *Parametric Route* (frozen macro weight layouts) and a highly reactive *Non-Parametric Route* (fast key-value structural override patches).
* **World-Model Information Recovery (WMIR):** Maximizes the right-sided derivative of log-likelihood recovery at the exact boundary of environmental failure ($t = 0^+$).

---

## 📊 Empirical Verification: The "Black Swan" Benchmark

Standard neural networks experience parameter saturation when environmental rules suddenly invert, resulting in a flat or slow recovery trajectory. RTI detects this structural breakdown instantly and hot-swaps active hypothesis rules.

### Performance Profile (WMIR Comparison)
Below is the empirical evaluation mapping the recovery velocity post-distribution shift:

![RTI Recovery Velocity](RTI_vs_Gradient_Recovery_Velocity.png)

*The RTI model achieves instant, optimal likelihood recovery ($\log P = -0.6931$) at $t=1$, while continuous gradient descent languishes due to activation function flattening.*

---

## 💻 Code Structure & Quick Start

The repository contains a self-contained evaluation harness mapping the **Inverted Physics Gridworld Challenge**, where environment rules radically reverse mid-run.

### Prerequisites
```bash
pip install numpy matplotlib
