# Slide Deck Outline
# Spend the Scan Time Wisely: Learned Averaging-Allocation for Low-Field MRI

> Export to PPTX/PDF using your preferred tool (PowerPoint, Google Slides, Keynote, LaTeX Beamer).
> Each slide has a title, bullet points, and the suggested figure/plot.

---

## Slide 1 — Title

**Title:** Spend the Scan Time Wisely:
Learning the Averages-vs-Coverage Allocation Jointly with Reconstruction for Low-Field MRI

**Subtitle:** Advanced Topics in Deep Learning for Medical Imaging

---

## Slide 2 — The Low-Field SNR Problem

- At 0.3 T, SNR is ~10–100× lower than clinical 3 T
- Remedy: **repeat acquisitions (NEX)** — but scan time is fixed
- **The tradeoff:** more averages on a few lines **vs** covering more k-space
- Prior work (Lau et al. 2023): use repetitions as fixed input to deep SR — time use not optimised

**Figure:** schematic of the averaging-vs-coverage tradeoff (k-space coverage × SNR trade-off diagram)

---

## Slide 3 — Problem Formulation

- Decision variable: **w_m ≥ 0** = averages per phase-encode line m
- Budget constraint: **Σ w_m = B** (fixed scan time)
  - w_m = 0 → line not acquired
  - w_m > 1 → averaged multiple times → noise ↓
- Measurement model: **ỹ_m = (Fx)_m + (σ_m / √(w_m+ε)) · η_m**
- **Novelty:** unify sampling and averaging in one learnable budget vector

**Figure:** diagram showing the noise model and the budget allocation

---

## Slide 4 — Method: Policy + Reconstructor

**Policy (learnable):**
- Logits a ∈ R^N
- **w = B · softmax(a/τ)** — budget-feasible, differentiable, temperature-annealed
- Noise profile σ_m calibrated from M4Raw inter-repetition variance

**Reconstructor (unrolled VarNet, K=6 steps):**
- **Precision-weighted DC:** x ← x − η·F^H[ρ ⊙ (Fx − ỹ)], where ρ_m = w_m/σ_m²
- CNN denoiser (U-Net) conditioned on ρ

**Joint training:** minimise L_img(R_θ(ỹ(w), ρ), x) w.r.t. a and θ simultaneously

**Figure:** architecture diagram — policy layer → measurement → unrolled recon → loss

---

## Slide 5 — Dataset: M4Raw

- **Lyu et al., Scientific Data 2023**
- 183 subjects × {T1w, T2w, FLAIR} × **2–3 repetitions** at 0.3 T
- 4-channel receive coil; fastMRI-format .h5 files
- **The repetitions let us:** (a) estimate real per-line σ_m, (b) physically realise w_m

**Figure:** sample M4Raw reconstruction at varying numbers of averaged repetitions

---

## Slide 6 — Baselines (5 comparators)

| # | Method | Allocation | Recon |
|---|--------|-----------|-------|
| 1 | Full NEX=1 IFFT | w_m = 1 ∀m | IFFT |
| 2 | Uniform averaging | w_m = B/N ∀m | VarNet |
| 3 | CS-VDA (fixed) | Center-biased polynomial | VarNet |
| 4 | Fixed Poisson-disc + VarNet | Binary VD mask | VarNet |
| 5 | LOUPE (learned, NEX=1) | Learned binary mask | VarNet |
| — | **Proposed** | **Learned w** | **VarNet** |

Key ablations: Proposed vs (5) isolates the **averaging axis**; vs (2)/(3) isolates **learned vs fixed**.

---

## Slide 7 — Main Results (E1)

**Figure (a):** PSNR / SSIM vs scan-time budget B/N for all methods

Key takeaways:
- Proposed > all baselines at iso-time (B = N)
- Gains largest at B < N (undersampling) where allocation choices matter most
- Proposed closely approaches LOUPE at B = N (high SNR); diverges at B < N (low SNR)

---

## Slide 8 — Low-SNR Advantage (E2)

**Figure (b):** PSNR / SSIM vs noise scale σ_scale (higher = lower SNR)

Key takeaways:
- At high SNR (σ → 0): all methods converge (averaging not needed)
- At low SNR (σ large): **Proposed > LOUPE** — the averaging axis provides significant gain
- This is exactly the regime where low-field MRI operates

---

## Slide 9 — The Signature Figure: Learned Allocation w* (E3)

**Figure (c):** w*_m vs k-space position at low SNR vs high SNR

Key takeaways:
- **Low SNR:** w* concentrates at k-space center (many averages) + peripheral lines dropped
  - Matches the CS-VDA analytical prediction — but **learned from data**
- **High SNR:** w* broadens toward near-uniform full coverage
  - The policy **recovers known physics** without being given an analytical prescription
- The **reconstruction-awareness** makes the learned allocation better than CS-VDA in practice

---

## Slide 10 — Real-Repetition Evaluation (E5)

- Integer rounding: w* → integers summing to B (largest-remainder method)
- Physically realised on M4Raw: w_m actual repetitions averaged per line (cap: max 3)
- Results reported separately from simulated-noise eval (honest about extrapolation)

**Figure (e):** bar chart — simulated vs real-reps eval for proposed method, side by side

---

## Slide 11 — Ablations (E4)

**Figure (f):** SSIM bars for:
- Proposed (full)
- w/o averaging axis (LOUPE baseline)
- w/o precision-weighted DC (unweighted DC)
- w/o entropy regulariser
- w/o temperature annealing

Key insight: each component contributes; the averaging axis + precision-weighted DC are most critical.

---

## Slide 12 — Discussion and Conclusions

**We showed:**
- Learned averaging allocation **outperforms** fixed CS-VDA and NEX=1 learned masks (LOUPE) at matched scan time
- Largest gains in the **low-SNR regime** (0.3 T)
- Learned policy **recovers the known SNR-dependent acquisition optimum from data**
- Physically realised on real M4Raw repetitions → deployable protocol

**Limitations:**
- Real-reps cap (R ≤ 3) limits realizable budgets; simulated mode extrapolates
- Single-contrast, single-coil (multi-contrast/coil: future work)

**Future directions:**
- Multi-contrast budget split (learn B_T1 + B_T2 + B_FLAIR)
- Hardware-in-the-loop optimisation
- Extension to 3D and non-Cartesian trajectories

---

## Slide 13 — References

1. Lau et al. Magn Reson Med 2023;90:400–416 *(base paper)*
2. Lyu et al. Scientific Data 2023;10:264 *(M4Raw dataset)*
3. Schoormans et al. arXiv:1909.01672 *(CS-VDA)*
4. arXiv:2411.06704 *(Accelerating low-field MRI)*
5. arXiv:2511.05735 *(k-space coverage for denoising)*
6. Bahadir et al. TCI 2020 *(LOUPE)*
7. Weiss et al. 2019 *(PILOT)*
8. Sriram et al. MICCAI 2020 *(E2E-VarNet)*
