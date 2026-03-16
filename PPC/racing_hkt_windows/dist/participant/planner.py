import numpy as np


def fivesect_cones(pts):
    dense = []
    for i in range(len(pts) - 1):
        dense.append(pts[i])
        dense.append(pts[i] * (4/5) + pts[i + 1] * (1/5))
        dense.append(pts[i] * (3/5) + pts[i + 1] * (2/5))
        dense.append(pts[i] * (2/5) + pts[i + 1] * (3/5))
        dense.append(pts[i] * (1/5) + pts[i + 1] * (4/5))
    dense.append(pts[-1])
    return np.array(dense)


def resample(pts, n):
    diffs    = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s        = np.concatenate([[0], np.cumsum(seg_lens)])
    s_new    = np.linspace(0, s[-1], n)
    x_new    = np.interp(s_new, s, pts[:, 0])
    y_new    = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x_new, y_new])


def direction_changes(pts):
    n      = len(pts)
    angles = np.zeros(n)
    for i in range(1, n - 1):
        v1 = pts[i]   - pts[i-1]
        v2 = pts[i+1] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        v1 = v1/n1; v2 = v2/n2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        dot   = np.dot(v1, v2)
        angles[i] = np.arctan2(cross, dot)
    for idx in [0, n-1]:
        prev = (idx - 1) % n
        nxt  = (idx + 1) % n
        v1 = pts[idx] - pts[prev]
        v2 = pts[nxt]  - pts[idx]
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            v1 = v1/n1; v2 = v2/n2
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            angles[idx] = np.arctan2(cross, np.dot(v1, v2))
    return angles


def plan(cones: list[dict]) -> list[dict]:

    left  = [c for c in cones if c["side"] == "left"]
    right = [c for c in cones if c["side"] == "right"]

    left.sort(key=lambda c: c["index"])
    right.sort(key=lambda c: c["index"])

    n = min(len(left), len(right))
    if n == 0:
        return []

    left_pts  = np.array([[c["x"], c["y"]] for c in left[:n]])
    right_pts = np.array([[c["x"], c["y"]] for c in right[:n]])

    print(f"original cones: left={len(left_pts)}, right={len(right_pts)}")

    # --- Step 1: direction changes on original midpoints ---
    raw_mid      = (left_pts + right_pts) / 2
    nm           = len(raw_mid)
    mid_angles   = direction_changes(raw_mid)
    left_angles  = direction_changes(left_pts)
    right_angles = direction_changes(right_pts)
    avg_abs      = (np.abs(left_angles) + np.abs(right_angles) + np.abs(mid_angles)) / 3

    # --- Step 2: apex detection ---
    valid     = avg_abs[avg_abs > 0.01]
    threshold = np.percentile(valid, 80)

    apexes = []
    for i in range(nm):
        prev = (i - 1) % nm
        nxt  = (i + 1) % nm
        if avg_abs[i] > threshold:
            if avg_abs[i] >= avg_abs[prev] and avg_abs[i] >= avg_abs[nxt]:
                apexes.append((i, mid_angles[i]))

    print(f"Apex points: {[(a, round(np.degrees(s),1)) for a,s in apexes]}")

    # --- Step 3: smooth turn_side ---
    ksize = 7;  sigma = 3.0
    kernel_raw = np.exp(-0.5 * (np.arange(ksize) - ksize//2)**2 / sigma**2)
    kernel_raw /= kernel_raw.sum()
    angles_smooth = np.convolve(mid_angles, kernel_raw, mode='same')
    turn_side     = np.sign(angles_smooth)
    turn_side[turn_side == 0] = 1.0

    # --- Step 4: build alpha at 31 points ---
    baseline   = 0.28
    apex_mag   = 0.45
    apex_sigma = 2.0

    alpha_31 = -turn_side * baseline

    for (a, angle_sign) in apexes:
        ts = np.sign(angle_sign)
        if ts == 0: ts = 1.0
        for i in range(nm):
            if turn_side[i] != ts:
                continue
            dist  = abs(i - a)
            gauss = np.exp(-0.5 * (dist / apex_sigma)**2)
            alpha_31[i] += ts * (baseline + apex_mag) * gauss

    alpha_31 = np.clip(alpha_31, -0.85, 0.85)

    # --- Step 5: build dense path ---
    left_dense  = fivesect_cones(left_pts)
    right_dense = fivesect_cones(right_pts)
    left_res    = resample(left_dense,  150)
    right_res   = resample(right_dense, 150)
    m           = 150

    mid        = (left_res + right_res) / 2
    half_width = np.linalg.norm(left_res - right_res, axis=1) / 2
    left_dir   = left_res  - mid
    right_dir  = right_res - mid

    # --- Step 6: interpolate alpha 31 → 150 by arc length ---
    diffs_raw   = np.diff(raw_mid, axis=0)
    s_raw       = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs_raw, axis=1))])
    s_raw_n     = s_raw / s_raw[-1]

    diffs_dense = np.diff(mid, axis=0)
    s_dense     = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs_dense, axis=1))])
    s_dense_n   = s_dense / s_dense[-1]

    alpha_150 = np.interp(s_dense_n, s_raw_n, alpha_31)

    # --- Step 7: compute racing line ---
    racing = np.zeros((m, 2))
    for i in range(m):
        if alpha_150[i] >= 0:
            racing[i] = mid[i] + alpha_150[i] * left_dir[i]
        else:
            racing[i] = mid[i] + abs(alpha_150[i]) * right_dir[i]

    # clamp to 85% of half_width
    for i in range(m):
        dist = np.linalg.norm(racing[i] - mid[i])
        maxd = 0.85 * half_width[i]
        if dist > maxd and dist > 1e-6:
            racing[i] = mid[i] + (racing[i] - mid[i]) * maxd / dist

    # --- Step 8: find switchover point ---
    # Only look AFTER the first apex — centerline is followed until then.
    # Switch when racing line and centerline are close AND aligned.
    if apexes:
        first_apex_31 = apexes[0][0]
        apex_s_norm   = s_raw_n[first_apex_31]
        apex_150      = int(np.searchsorted(s_dense_n, apex_s_norm))
    else:
        apex_150 = 0

    switchover = m // 2   # fallback

    for i in range(apex_150 + 1, m - 1):
        dist_to_mid = np.linalg.norm(racing[i] - mid[i])

        dir_mid  = mid[i+1]    - mid[i]
        dir_race = racing[i+1] - racing[i]
        n1 = np.linalg.norm(dir_mid)
        n2 = np.linalg.norm(dir_race)
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        alignment = np.dot(dir_mid / n1, dir_race / n2)

        if dist_to_mid < 0.5 and alignment > 0.97:
            switchover = i
            break

    print(f"First apex at dense wp={apex_150}, switchover at wp={switchover} / {m}")

    # --- Step 9: build final path ---
    # centerline up to switchover, racing line after
    final = np.zeros((m, 2))
    for i in range(m):
        if i < switchover:
            final[i] = mid[i]
        else:
            final[i] = racing[i]

    # close the loop
    final[-1] = final[0]

    print(f"final waypoints: {m}")
    return [{"x": float(p[0]), "y": float(p[1])} for p in final]
