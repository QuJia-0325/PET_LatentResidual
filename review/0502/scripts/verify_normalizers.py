"""Standalone verification of sigma-normalize divisors.

Run BEFORE applying patches to confirm the math used in:
  - SIGMA_NORMALIZE_ABLATION_PLAN.md §2.1, §2.2
  - configs/B_sanity.yaml, configs/C_uniform.yaml, configs/D_closed_form.yaml step_weights

Usage:
    cd review/0502
    python scripts/verify_normalizers.py

The script verifies that the `preserve_v6_sum` normalizer mode produces SGD-equivalent
gradients between A_control (no σ-norm, V6 weights) and B_sanity (σ-norm + rescaled
weights) under the weighted-AVERAGE loss form used by rollout_first_hop.py:

    loss_total = (stacked * w).sum() / w.sum()

A == B requires BOTH numerator equality AND denominator equality, which is achieved by:

    n_j         = rho_j × Σw_v6 / Σ(w_v6 × rho)        # weighted-mean reference
    w_B_yaml    = w_v6 × n     ⇒ Σw_B = Σw_v6  (sum conservation)
    grad_total  = Σ rho_j × w_v6_j × ‖δv_j‖² / Σw_v6  (identical to A)

Expected output (matches yaml configs to 4 decimals; FP64 residual ~2.7e-20):
    rho_j (sigma*dt)^2          = [8.3533e-04, 2.1697e-04, 1.3514e-04, 1.1183e-04]
    weighted_mean(rho, w_v6)    = 2.3323e-04
    normalizer (preserve_v6_sum)= [3.5816, 0.9303, 0.5794, 0.4795]
    Config B sanity step_weights= [1.7908, 1.8606, 0.8691, 0.4795]   sum=5.0000
    Config C uniform step_weights= [1.2500, 1.2500, 1.2500, 1.2500]  sum=5.0000
    Config D closedform weights = [3.5795, 0.7910, 0.4149, 0.2146]   sum=5.0000
    Σ(w_v6) baseline            = 5.0000
    Σ(w_yaml) for all configs   = 5.0000  (effective lambda_roll matched across A/B/C/D)
    weighted_mean(n, w_v6) (=1.0)        : 1.000000
    max |contribA - contribB| per hop    : 2.710505e-20  (FP64 numerical noise floor)
    |sum(w_B) - sum(w_v6)|               : 0.000000e+00
"""
import statistics


def main() -> None:
    # V6 config values (must match
    #  configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml).
    pair_v_std = [0.009634, 0.002946, 0.000775, 0.000141]   # first_hop.pair_v_std
    rollout_times = [2.0, 5.0, 10.0, 25.0, 100.0]            # data.t_map for D50→NORMAL
    v6_step_weights = [0.5, 2.0, 1.5, 1.0]                   # training.rollout.step_weights (anchor)
    closedform_raw = [3.35, 2.85, 2.4, 1.5]                  # §18.3.3 closed-form Grönwall

    dts = [rollout_times[j + 1] - rollout_times[j] for j in range(len(rollout_times) - 1)]
    rho = [(pair_v_std[j] * dts[j]) ** 2 for j in range(len(dts))]
    median_rho = statistics.median(rho)
    sum_v6 = sum(v6_step_weights)
    weighted_mean_rho = sum(v6_step_weights[j] * rho[j] for j in range(len(rho))) / sum_v6

    # Two normalizer modes: legacy (median) and recommended (preserve_v6_sum).
    norm_median = [r / median_rho for r in rho]
    norm_preserve = [r / weighted_mean_rho for r in rho]

    # Step weights using preserve_v6_sum normalizer:
    config_b = [w * n for w, n in zip(v6_step_weights, norm_preserve)]
    # C uniform: rescale [1,1,1,1] to sum = sum_v6 = 5.0  →  [1.25, 1.25, 1.25, 1.25]
    config_c = [sum_v6 / len(v6_step_weights)] * len(v6_step_weights)
    # D closed-form: w_D_pre = closed × n, then rescale so Σw_D = sum_v6 (preserve effective lambda_roll)
    w_d_pre = [w * n for w, n in zip(closedform_raw, norm_preserve)]
    sum_d_pre = sum(w_d_pre)
    config_d = [w * sum_v6 / sum_d_pre for w in w_d_pre]

    sum_b = sum(config_b)
    sum_c = sum(config_c)
    sum_d = sum(config_d)

    # Velocity-space normalized distributions (peak=1) — what each condition's gradient
    # direction looks like in normalized-velocity space, after σ-norm cancels rho.
    # contrib_j ∝ w_yaml_j (because w_yaml_j × rho_j / n_j = w_yaml_j × weighted_mean_rho).
    v6_vel_pre_norm = [w * r for w, r in zip(v6_step_weights, rho)]   # A path: no σ-norm
    v6_vel_peak = max(v6_vel_pre_norm)
    v6_vel = [e / v6_vel_peak for e in v6_vel_pre_norm]

    cf_vel_pre_norm = [w * r for w, r in zip(closedform_raw, rho)]    # closed-form unweighted
    cf_vel_peak = max(cf_vel_pre_norm)
    cf_vel = [e / cf_vel_peak for e in cf_vel_pre_norm]

    # Sanity check #1: weighted-mean of normalizer w.r.t. v6 weights should equal 1.0
    n_w_mean = sum(v6_step_weights[j] * norm_preserve[j] for j in range(len(rho))) / sum_v6
    # Sanity check #2: SGD-direction equivalence of (A: raw, w_v6) vs (B: σ-norm, w_v6×n)
    # contribution per hop: A: w_v6 × rho ; B: (w_v6 × n) × (rho / n) = w_v6 × rho ✓
    contribA = [v6_step_weights[j] * rho[j] for j in range(len(rho))]
    contribB = [config_b[j] * (rho[j] / norm_preserve[j]) for j in range(len(rho))]
    contrib_max_err = max(abs(a - b) for a, b in zip(contribA, contribB))
    sum_err_b = abs(sum_b - sum_v6)

    print("=== Sigma-normalize divisors verification (preserve_v6_sum mode) ===")
    print(f"dt                       = {dts}")
    print(f"sigma                    = {pair_v_std}")
    print(f"rho = (sigma*dt)^2       = [{', '.join(f'{r:.4e}' for r in rho)}]")
    print(f"median(rho)              = {median_rho:.4e}")
    print(f"weighted_mean(rho, w_v6) = {weighted_mean_rho:.4e}")
    print(f"normalizer / median      = [{', '.join(f'{n:.4f}' for n in norm_median)}]   (legacy)")
    print(f"normalizer / w_mean(rho) = [{', '.join(f'{n:.4f}' for n in norm_preserve)}]   (preserve_v6_sum)")
    print()
    print("=== Step weights (preserve_v6_sum mode, all configs Σw ≈ Σw_v6 = 5.0) ===")
    print(f"A_control  step_weights  = [{', '.join(f'{w:.4f}' for w in v6_step_weights)}]   sum={sum_v6:.4f}")
    print(f"B_sanity   step_weights  = [{', '.join(f'{w:.4f}' for w in config_b)}]   sum={sum_b:.4f}")
    print(f"C_uniform  step_weights  = [{', '.join(f'{w:.4f}' for w in config_c)}]   sum={sum_c:.4f}")
    print(f"D_closed   step_weights  = [{', '.join(f'{w:.4f}' for w in config_d)}]   sum={sum_d:.4f}")
    print()
    print("=== Velocity-space distributions (raw rollout × step_weights, peak=1) ===")
    print(f"V6  vel-space            = [{', '.join(f'{v:.3f}' for v in v6_vel)}]")
    print(f"closed-form vel-space    = [{', '.join(f'{v:.3f}' for v in cf_vel)}]")
    print()
    print("=== Mathematical identity checks ===")
    print(f"weighted_mean(n, w_v6) (must=1.0)            : {n_w_mean:.6f}   "
          f"{'OK' if abs(n_w_mean - 1.0) < 1e-9 else 'FAIL'}")
    print(f"max |contribA - contribB| per hop (must≈0)   : {contrib_max_err:.6e}   "
          f"{'OK' if contrib_max_err < 1e-9 else 'FAIL'}")
    print(f"|sum(w_B) - sum(w_v6)|             (must≈0)  : {sum_err_b:.6e}   "
          f"{'OK' if sum_err_b < 1e-3 else 'FAIL'}")
    print()
    print("=== Per-hop relative magnitude (effective contribution to total) ===")
    contribA_norm = [c / sum(contribA) for c in contribA]
    contribC = [config_c[j] * (rho[j] / norm_preserve[j]) for j in range(len(rho))]
    contribC_norm = [c / sum(contribC) for c in contribC]
    contribD = [config_d[j] * (rho[j] / norm_preserve[j]) for j in range(len(rho))]
    contribD_norm = [c / sum(contribD) for c in contribD]
    print(f"A == B (no σ-norm) hop fraction = "
          f"[{', '.join(f'{c:.3f}' for c in contribA_norm)}]")
    print(f"C uniform hop fraction          = "
          f"[{', '.join(f'{c:.3f}' for c in contribC_norm)}]   (must be ~equal)")
    print(f"D closed-form hop fraction      = "
          f"[{', '.join(f'{c:.3f}' for c in contribD_norm)}]   (peaks at hop0 / D50→D20)")


if __name__ == "__main__":
    main()
