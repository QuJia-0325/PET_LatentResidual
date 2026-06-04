# S2.b Pilot Correlation Gate

日期: 2026-06-04 16:34:51 +0800

## Verdict

- Gate: PASS
- Decision: `continue_to_S2a`
- Rule: PASS if overall Spearman(q,e)-Spearman(l,e) >= 0.15 and log-log Pearson(q,e) > 0.5
- Spearman advantage `corr(q,e)-corr(l,e)`: 0.3900
- Log-Pearson advantage `corr(log q,log e)-corr(log l,log e)`: 0.1049

## Overall Correlations

| metric | value |
|---|---:|
| `n` | 32.000000 |
| `pearson_q_e` | 0.994476 |
| `pearson_l_e` | 0.885183 |
| `spearman_q_e` | 0.997434 |
| `spearman_l_e` | 0.607405 |
| `pearson_log_q_log_e` | 0.999758 |
| `pearson_log_l_log_e` | 0.894845 |
| `spearman_advantage_q_minus_l` | 0.390029 |
| `pearson_log_advantage_q_minus_l` | 0.104912 |

## Group Correlations

| group | n | Pearson(q,e) | Pearson(l,e) | Spearman(q,e) | Spearman(l,e) | Spearman adv |
|---|---:|---:|---:|---:|---:|---:|
| `overall` | 32 | 0.9945 | 0.8852 | 0.9974 | 0.6074 | 0.3900 |
| `model=V13` | 16 | 0.9932 | 0.9194 | 1.0000 | 0.6235 | 0.3765 |
| `model=A4` | 16 | 0.9961 | 0.8462 | 0.9971 | 0.5941 | 0.4029 |
| `dst=D20` | 16 | 0.9925 | 0.8739 | 0.9912 | 0.6088 | 0.3824 |
| `dst=NORMAL` | 16 | 0.9992 | 0.4392 | 0.9882 | 0.5353 | 0.4529 |

## Protocol

- Rows: `32` = 2 models × 16 samples.
- `q = δᵀMδ = ||J_G δ||²` computed by matrix-free JVP in clip3 SUV [0,3] domain.
- `e = ||G(z_gt + δ) - G(z_gt)||²` is the true nonlinear decode-domain error in the same clip3 domain.
- `l = ||δ||²` is the latent-space baseline.
- Single-step protocol uses GT source latent; hop0 receives GT D50 image conditioning; NORMAL uses GT D4 source latent.

## Artifacts

- JSON: `review/0603/server/S2b_pilot_correlation_20260604.json`
- CSV: `review/0603/server/S2b_pilot_correlation_20260604.csv`
- Report: `review/0603/server/S2b_pilot_correlation_20260604.md`
