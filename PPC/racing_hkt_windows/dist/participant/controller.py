import numpy as np

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
current_waypoint = 0
_last_path_len   = 0
_tangents = None
_radii    = None
_speeds   = None


def circumcircle(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    D = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(D) < 1e-8:
        return None, np.inf

    ux = ((ax**2 + ay**2)*(by - cy) +
          (bx**2 + by**2)*(cy - ay) +
          (cx**2 + cy**2)*(ay - by)) / D

    uy = ((ax**2 + ay**2)*(cx - bx) +
          (bx**2 + by**2)*(ax - cx) +
          (cx**2 + cy**2)*(bx - ax)) / D

    center = np.array([ux, uy])
    R      = np.linalg.norm(p2 - center)
    return center, R


def tangent_at_p2(p1, p2, p3, center):
    if center is None:
        t = np.array(p3) - np.array(p1)
        n = np.linalg.norm(t)
        return t / n if n > 1e-8 else np.array([1.0, 0.0])

    radial    = np.array(p2) - center
    t1        = np.array([-radial[1],  radial[0]])
    t2        = np.array([ radial[1], -radial[0]])
    track_dir = np.array(p3) - np.array(p1)
    tangent   = t1 if np.dot(t1, track_dir) >= 0 else t2
    n = np.linalg.norm(tangent)
    return tangent / n if n > 1e-8 else np.array([1.0, 0.0])


def precompute(path):
    pts = np.array([[p["x"], p["y"]] for p in path])
    n   = len(pts)

    tangents = np.zeros((n, 2))
    radii    = np.full(n, np.inf)

    # Sliding window: (0,1,2), (1,2,3), (2,3,4) ...
    # With 150 points from cone-interpolated midpoints, consecutive
    # triplets are no longer collinear — they follow actual curve geometry.
    for i in range(1, n - 1):
        p1 = pts[i - 1]
        p2 = pts[i]
        p3 = pts[i + 1]

        d1 = np.linalg.norm(p2 - p1)
        d2 = np.linalg.norm(p3 - p2)
        if d1 < 0.05 or d2 < 0.05:
            radii[i]    = radii[i - 1]
            tangents[i] = tangents[i - 1]
            continue

        center, R   = circumcircle(p1, p2, p3)
        tangents[i] = tangent_at_p2(p1, p2, p3, center)
        radii[i]    = R

    # fill edges
    tangents[0]  = tangents[1];   radii[0]  = radii[1]
    tangents[-1] = tangents[-2];  radii[-1] = radii[-2]

    # --- Speed profile: v = sqrt(A_LAT * R) ---
    A_LAT = 21.0
    V_MIN = 2.0
    V_MAX = 50.0

    speeds = np.where(
        radii > 1000,
        V_MAX,
        np.clip(np.sqrt(A_LAT * radii), V_MIN, V_MAX)
    )

    # --- Backward braking pass ---
    A_BRAKE = 9.0
    for i in range(n - 2, -1, -1):
        ds        = np.linalg.norm(pts[i+1] - pts[i])
        speeds[i] = min(speeds[i], np.sqrt(speeds[i+1]**2 + 2*A_BRAKE*ds))

    # --- First apex speed reduction only ---
    first_apex = None
    for i in range(1, n - 1):
        if radii[i] < radii[i-1] and radii[i] < radii[i+1] and radii[i] < 1000:
            first_apex = i
            break

    if first_apex is not None:
        apex_v     = np.clip(np.sqrt(A_LAT * radii[first_apex]), V_MIN, V_MAX)
        buffer_wps = 15
        free_after = 3

        for i in range(max(0, first_apex - buffer_wps), min(n, first_apex + free_after)):
            dist = i - first_apex
            if dist <= 0:
                t = (dist + buffer_wps) / buffer_wps
                speeds[i] = min(speeds[i], speeds[i] * (1-t) + apex_v * t)
            else:
                speeds[i] = min(speeds[i], apex_v)

        print(f"First apex at wp={first_apex}, R={radii[first_apex]:.2f}, apex_v={apex_v:.2f}m/s")

    # debug
    finite = radii < 1000
    if np.any(finite):
        min_r   = np.min(radii[finite])
        min_idx = np.argmin(radii)
        print(f"sharpest corner: R={min_r:.2f}m at wp={min_idx}, target_v={np.sqrt(A_LAT*min_r):.2f}m/s")
    print(f"max spd: {np.max(speeds):.2f}  min spd: {np.min(speeds):.2f}  avg: {np.mean(speeds):.2f}")

    return tangents, radii, speeds


def control(path, state, cmd_feedback, step):
    global current_waypoint, _last_path_len
    global _tangents, _radii, _speeds

    if len(path) != _last_path_len:
        current_waypoint = 0
        _last_path_len   = len(path)
        if len(path) >= 3:
            _tangents, _radii, _speeds = precompute(path)
        else:
            _tangents = _radii = _speeds = None

    if len(path) < 2 or _tangents is None:
        return 0.0, 0.0, 0.0

    car   = np.array([state["x"], state["y"]])
    speed = np.sqrt(state["vx"]**2 + state["vy"]**2)
    pts   = np.array([[p["x"], p["y"]] for p in path])
    n     = len(pts)

    # advance waypoint
    while current_waypoint < n - 2:
        p1      = pts[current_waypoint]
        p2      = pts[current_waypoint + 1]
        seg     = p2 - p1
        seg_len = np.linalg.norm(seg) + 1e-6
        proj    = np.dot(car - p1, seg / seg_len)
        if proj < seg_len * 0.5:
            break
        current_waypoint += 1

    idx = min(current_waypoint, n - 2)

    # signed cross-track error (positive = car LEFT of path)
    p1      = pts[idx]
    p2      = pts[idx + 1]
    seg     = p2 - p1
    seg_len = np.linalg.norm(seg) + 1e-6
    seg_dir = seg / seg_len
    car_vec = car - p1
    cross_track = seg_dir[0]*car_vec[1] - seg_dir[1]*car_vec[0]

    # heading error from circumcircle tangent
    path_heading  = np.arctan2(_tangents[idx][1], _tangents[idx][0])
    heading_error = path_heading - state["yaw"]
    heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

    # Stanley steering
    k     = 0.8
    steer = heading_error - np.arctan2(k * cross_track, speed + 1e-6)
    steer = np.clip(steer, -0.5, 0.5)

    # speed control
    target_speed = _speeds[idx]
    speed_error  = target_speed - speed

    if speed_error > 0:
        throttle = np.clip(0.4 + 0.06 * speed_error, 0.0, 1.0)
        brake    = 0.0
    else:
        throttle = 0.0
        brake    = np.clip(-0.5 * speed_error, 0.0, 1.0)

    if step % 20 == 0:
        print(f"idx={idx}  R={_radii[idx]:.1f}  cross={cross_track:.3f}  steer={steer:.3f}  spd={speed:.2f}  tgt={target_speed:.2f}")

    return float(throttle), float(steer), float(brake)
