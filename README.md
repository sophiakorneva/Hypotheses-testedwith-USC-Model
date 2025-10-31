# Hypothesis Testing for the Ultra-Slow Component Model (USC-Model)

**Author:** Sofia Korneva  
**Affiliation:** Federal Medical Biophysical Center (FMBA of Russia) & Lomonosov Moscow State University  

---

## Overview
This repository contains scripts for quantitative testing of mechanistic hypotheses explaining the differences between γ- and neutron-induced DNA damage responses, based on the **Ultra-Slow Component Model (USC-Model, Model C⁺)**.

The model extends the Bi-Component Repair Model (BCRM) by introducing an ultraslow fraction of complex DNA double-strand breaks (DSBs).  
Here, three main hypotheses (H₁–H₃) were tested using **ΔAIC model comparison** and **bootstrap confidence intervals**.

---

## Hypotheses

| ID | Hypothesis | Biological meaning | Tested parameters |
|----|-------------|-------------------|-------------------|
| H₁ (k_rs↑) | For neutrons, simple DSBs repair faster (steeper early γH2AX decay 0.5–2 h). | NHEJ acceleration | k_rs |
| H₂ (w_dc↑) | One complex DSB under neutrons induces a stronger ATM/γH2AX signal. | Enhanced signaling strength | w_dc |
| H₃ (π↑, k_cu↓) | Neutron irradiation produces a larger fraction of ultraslow breaks (π ↑) and longer retention in the complex pool (k_cu ↓). | Retention of complex/ultraslow DSBs | π, k_cu |

---

## Methodology

- Optimization performed with `scipy.optimize.least_squares` (TRF algorithm).  
- Model comparison via **ΔAIC = AIC_alt − AIC_base**.  
- Bootstrap resampling (500–1000 iterations) used to estimate confidence intervals for parameter differences.  
- Separate fits conducted for:
  - Early window (0.5–2 h) — tests H₁, H₂  
  - Full window (0.5–24 h) — tests H₃

---

## Results

### Early window (0.5–2 h)
**H₁** not supported:  
k_rs(γ) ≈ 1.35–1.40 > k_rs(n) ≈ 1.08, ΔAIC(M1 vs M2) ≈ 0  
Bootstrap median Δk_rs = +0.29, 95% CI [−0.75; +0.28]; P(Δ>0) ≈ 0.13  

**H₂** shows a weak positive tendency:  
Δw_dc ≈ +1.7 (wide CI including 0) — compatible with stronger per-break signaling but not statistically significant.  

### Full window (0.5–24 h)
**H₃** strongly supported:  
ΔAIC ≈ 8.3 in favor of the model with separate π and k_cu for γ and n.  
Δπ ≈ +0.28 (95% CI [−0.16; −0.52]), Δk_cu ≈ −0.33 (95% CI [−0.94; +0.63]).  

**Conclusion:**  
Differences between γ- and neutron-induced damage are primarily due to **retention and delayed migration of complex DSBs**, not due to faster repair of simple DSBs.

---

## Statistical summary

| Hypothesis | Supported? | Key metrics | Interpretation |
|-------------|-------------|--------------|----------------|
| H₁ (k_rs↑) | No | ΔAIC ≈ 0 | No evidence for faster repair under neutrons |
| H₂ (w_dc↑) | Partial trend | Δw_dc ≈ +1.7 | Slightly stronger ATM/γH2AX activation, not significant |
| H₃ (π↑, k_cu↓) | Yes | ΔAIC ≈ 8.3 | More ultraslow breaks and longer retention after neutrons |

---

## Implementation
The repository includes the script **`hypotheses_test.py`**, derived from experimental data fits performed in the USC-Model project.

**Main functions:**
```python
fit_model()
compare_models_AIC()
bootstrap_parameters()
plot_distributions()
