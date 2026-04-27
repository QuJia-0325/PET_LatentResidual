#!/bin/bash
# ================================================================
# V6 Pilot Go/No-Go 自动检查
# ================================================================
# 读取 V6 训练的 metrics JSONL，检查各阶段 Go/No-Go 条件
#
# 用法: bash review/0428/operator/02_v6_monitor.sh
# ================================================================
set -euo pipefail

PYTHON="/home/qujiaxiang/.conda/envs/rae/bin/python"
V6_DIR="/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first"
JSONL="${V6_DIR}/metrics.jsonl"

if [ ! -f "${JSONL}" ]; then
    echo "ERROR: ${JSONL} 不存在，训练尚未启动或未产生日志"
    exit 1
fi

${PYTHON} << 'PYEOF'
import json, sys

jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl"

train_events = []
with open(jsonl_path) as f:
    for line in f:
        d = json.loads(line)
        if d.get("event") == "train":
            train_events.append(d)

if not train_events:
    print("No train events found.")
    sys.exit(0)

last_step = train_events[-1]["step"]
print(f"=== V6 Monitor: {len(train_events)} train events, last step = {last_step} ===")
print()

# Helper: find events near target step
def near(target, tolerance=500):
    candidates = [e for e in train_events if abs(e["step"] - target) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x["step"] - target))

# +1K check
if last_step >= 1000:
    e = near(1000)
    if e:
        pair_raw = e.get("pair", 0)
        has_nan = not all(
            isinstance(e.get(k), (int, float)) and e.get(k) == e.get(k)
            for k in ["pair", "roll", "img", "loss"]
        )
        status = "PASS" if pair_raw < 5e-4 and not has_nan else "FAIL"
        print(f"[+1K] pair_raw={pair_raw:.4e}, NaN={has_nan} → {status}")
    print()

# +10K check
if last_step >= 10000:
    e = near(10000)
    if e:
        pf = e.get("pair_frac", 0)
        plw = e.get("pair_loss_weight", 1.0)
        status = "PASS" if pf > 0.70 else "FAIL"
        print(f"[+10K] pair_frac={pf:.3f}, pair_loss_weight={plw} → {status}")
    print()

# +50K check
if last_step >= 50000:
    e = near(50000)
    if e:
        imf = e.get("img_frac", 0)
        status = "PASS" if imf < 0.30 else "FAIL"
        print(f"[+50K] img_frac={imf:.3f} → {status}")
    print()

# +130K check
if last_step >= 130000:
    e = near(130000)
    if e:
        imf = e.get("img_frac", 0)
        img_raw = e.get("img", 0)
        status = "PASS" if imf < 0.30 else "WARN"
        print(f"[+130K] img_frac={imf:.3f}, img_raw={img_raw:.4e} → {status}")
        # Check img_raw trend (100K-130K)
        events_100_130 = [ev for ev in train_events if 100000 <= ev["step"] <= 130000 and "img" in ev]
        if len(events_100_130) >= 10:
            first_half = [ev["img"] for ev in events_100_130[:len(events_100_130)//2]]
            second_half = [ev["img"] for ev in events_100_130[len(events_100_130)//2:]]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            trend = "RISING" if avg_second > avg_first * 1.3 else "STABLE"
            print(f"  img_raw trend 100K-130K: first_half={avg_first:.4e}, second_half={avg_second:.4e} → {trend}")
    print()

# +150K check
if last_step >= 150000:
    e = near(150000)
    if e:
        pf = e.get("pair_frac", 0)
        rf = e.get("roll_frac", 0)
        transport = pf + rf
        status = "PASS" if transport >= 0.60 else "WARN"
        print(f"[+150K] transport_frac={transport:.3f} (pair={pf:.3f} + roll={rf:.3f}) → {status}")
    print()

# Summary
print("=== Current progress ===")
last = train_events[-1]
print(f"step={last['step']}, loss={last.get('loss', '?'):.6e}")
print(f"pair_frac={last.get('pair_frac', '?'):.3f}, roll_frac={last.get('roll_frac', '?'):.3f}, img_frac={last.get('img_frac', '?'):.3f}")
print(f"pair_loss_weight={last.get('pair_loss_weight', '?')}")
print(f"lambda_roll={last.get('lambda_roll', '?'):.4f}, lambda_img={last.get('lambda_img', '?'):.4f}, alpha={last.get('alpha', '?'):.3f}")
PYEOF
