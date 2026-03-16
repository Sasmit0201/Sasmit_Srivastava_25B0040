import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   # radians (0 = east)

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    """
    Pair each blue cone with its nearest yellow cone, take the midpoint,
    then sort CLOCKWISE around the track centroid so pure-pursuit drives CW.
    """
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]   # descending angle = clockwise


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   # metres – sensor visibility radius
NOISE_STD    = 0.20   # metres – measurement noise std-dev
WHEELBASE    = 3.0    # metres – bicycle model wheelbase
DT           = 0.1    # seconds – time step
SPEED        = 7.0    # m/s
LOOKAHEAD    = 5.5    # pure-pursuit lookahead distance (m)
N_FRAMES     = 130    # ≈ one full lap


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    """Compute steering angle (rad) to follow *path* via pure-pursuit."""
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       # fallback lookahead
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))


def local_to_global(local_pts: np.ndarray,
                    pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate + translate points from the car's local frame to world frame."""
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       # local → world rotation
    return (R @ local_pts.T).T + pos


def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left).
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)


def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b,
               zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y,
               zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7,
               zorder=2, label="Start gate")


def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)


def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   # (2,) float64
        self.heading = CAR_START_HEADING      # radians

    def data_association(self, measurements, current_map):
        raise NotImplementedError

    def localization(self, velocity, steering):
        raise NotImplementedError

    def mapping(self, measurements):
        raise NotImplementedError


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map  = []                    # list of np.ndarray (2,)
        # Internal state exposed for visualisation
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)

    # ------------------------------------------------------------------
    def localization(self, velocity, steering):
        """
        EKF predict + update — bicycle kinematic model.

        State vector: x = [px, py, ψ]

        ── Predict ──────────────────────────────────────────────────────────
            px  +=  v · cos(ψ) · dt
            py  +=  v · sin(ψ) · dt
            ψ   +=  (v / L) · tan(δ) · dt
            P    =  F · P · Fᵀ + Q

        ── Update ───────────────────────────────────────────────────────────
        For each visible cone that can be confidently matched to a map cone
        (nearest-neighbour with tight gate), apply the EKF correction:

            innovation  v  =  z_global - landmark_pos
            S              =  H · P · Hᵀ + R          (innovation covariance)
            K              =  P · Hᵀ · S⁻¹ · α        (dampened Kalman gain)
            [px, py]      +=  K[:2] · v
            ψ             +=  K[2]  · (v projected onto heading)
            P              =  (I - K · H) · P

        α is a dampening factor that prevents P from collapsing when
        associations are noisy — critical since we have no JCBB here.

        P is clamped to a minimum diagonal to stop it shrinking so far
        that the system becomes overconfident and stops accepting corrections.
        """
        # ── Lazy-init ──────────────────────────────────────────────────────
        if not hasattr(self, 'P'):
            self.P = np.diag([1.0, 1.0, 0.1])

        # ── Process noise Q (scales with velocity) ─────────────────────────
        pos_noise     = (0.05 * velocity * DT) ** 2
        heading_noise = (0.01 * velocity * DT) ** 2
        Q = np.diag([pos_noise, pos_noise, heading_noise])

        # ── Jacobian F ─────────────────────────────────────────────────────
        F = np.array([
            [1.0, 0.0, -velocity * np.sin(self.heading) * DT],
            [0.0, 1.0,  velocity * np.cos(self.heading) * DT],
            [0.0, 0.0,  1.0]
        ])

        # ── Predict: state ─────────────────────────────────────────────────
        self.pos[0]  += velocity * np.cos(self.heading) * DT
        self.pos[1]  += velocity * np.sin(self.heading) * DT
        self.heading  = angle_wrap(
            self.heading + (velocity / WHEELBASE) * np.tan(steering) * DT
        )

        # ── Predict: covariance ────────────────────────────────────────────
        self.P = F @ self.P @ F.T + Q

        # ── Update step ────────────────────────────────────────────────────
        # Get current noisy measurements in local frame
        meas = get_measurements(self.pos, self.heading)
        if len(meas) == 0:
            return

        # Project to global frame
        c, s   = np.cos(self.heading), np.sin(self.heading)
        R_l2g  = np.array([[c, -s], [s, c]])
        gm     = (R_l2g @ meas.T).T + self.pos

        # Observation model: H (2×3), R_obs noise covariance
        H      = np.array([[-1.0, 0.0, 0.0],
                            [ 0.0,-1.0, 0.0]])
        R_obs  = (NOISE_STD ** 2) * np.eye(2)

        # Dampening factor — keeps gain conservative without JCBB.
        # 0.15 means each update moves the estimate only 15% of what a
        # full Kalman correction would — prevents P from collapsing on
        # noisy or incorrectly matched observations.
        ALPHA  = 0.15

        # Minimum allowed P diagonal — stops the filter becoming so
        # confident it ignores future corrections entirely
        P_MIN  = np.diag([0.1, 0.1, 0.01])

        # Tight nearest-neighbour gate: only update on unambiguous matches
        # gate = 1.5m — tighter than the association gate so only clear
        # cone sightings drive the position correction
        UPDATE_GATE = 1.5 * NOISE_STD

        for z_global in gm:
            # Find nearest map cone
            dists   = np.linalg.norm(MAP_CONES - z_global, axis=1)
            nearest = int(np.argmin(dists))

            if dists[nearest] > UPDATE_GATE:
                continue   # ambiguous match — skip to avoid corrupting P

            landmark = MAP_CONES[nearest]

            # Innovation: difference between observed and expected position
            innovation = z_global - landmark

            # Innovation covariance and Kalman gain (dampened)
            S   = H @ self.P @ H.T + R_obs
            K   = (self.P @ H.T @ np.linalg.inv(S)) * ALPHA

            # State correction — only update position, not heading directly
            # (heading correction from position-only observations is weak)
            correction  = K @ innovation
            self.pos[0] += correction[0]
            self.pos[1] += correction[1]
            self.heading = angle_wrap(self.heading + correction[2])

            # Covariance update — Joseph form for numerical stability
            I_KH      = np.eye(3) - K @ H
            self.P    = I_KH @ self.P @ I_KH.T + K @ R_obs @ K.T

        # Clamp P to minimum — never let it shrink below P_MIN
        self.P = np.maximum(self.P, P_MIN)


# ── Problem 2 – Localization ───────────────────────────────────────────────────
def make_problem2():
    """
    Visualise EKF predict-step localization: the magenta trail is the car's
    estimated trajectory. Covariance ellipse shows growing pose uncertainty
    between correction steps.
    """
    sol     = Solution()
    path_x  = [float(sol.pos[0])]
    path_y  = [float(sol.pos[1])]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 2 – Localization  (EKF Predict Step)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        sol.localization(SPEED, steer)
        path_x.append(float(sol.pos[0]))
        path_y.append(float(sol.pos[1]))

        draw_track(ax)
        ax.plot(path_x, path_y, color="magenta", lw=2.0,
                alpha=0.85, zorder=4, label="EKF path")
        draw_car(ax, sol.pos, sol.heading)
        p_trace = np.trace(sol.P) if hasattr(sol, 'P') else 0.0
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"pos=({sol.pos[0]:.1f}, {sol.pos[1]:.1f})  "
            f"ψ={np.degrees(sol.heading):.1f}°  "
            f"tr(P)={p_trace:.3f}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  "
          f"heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 1 animation window …")

    # Keep references to prevent garbage collection of FuncAnimation objects.
    fig2, ani2 = make_problem2()

    plt.show()
