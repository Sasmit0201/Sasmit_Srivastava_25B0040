import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Centerline ─────────────────────────────────────────────────────────
def _build_centerline():

    center = np.mean(MAP_CONES, axis=0)

    D = distance.cdist(BLUE_CONES, YELLOW_CONES)

    mids = np.array([
        (BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
        for i in range(len(BLUE_CONES))
    ])

    angles = np.arctan2(mids[:,1]-center[1], mids[:,0]-center[0])

    return mids[np.argsort(angles)[::-1]]


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0
NOISE_STD = 1.0
WHEELBASE = 3.0
DT = 0.1
SPEED = 10.0
LOOKAHEAD = 5.5
N_FRAMES = 130


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos, heading, path):

    dists = np.linalg.norm(path-pos, axis=1)
    nearest = int(np.argmin(dists))
    n = len(path)

    target = path[(nearest+5)%n]

    for k in range(nearest, nearest+n):
        pt = path[k % n]
        if np.linalg.norm(pt-pos) >= LOOKAHEAD:
            target = pt
            break

    alpha = angle_wrap(np.arctan2(target[1]-pos[1],target[0]-pos[0]) - heading)

    steer = np.arctan2(2.0*WHEELBASE*np.sin(alpha), LOOKAHEAD)

    return float(np.clip(steer,-0.6,0.6))


def local_to_global(local_pts,pos,heading):

    c,s = np.cos(heading),np.sin(heading)
    R = np.array([[c,-s],[s,c]])

    return (R @ local_pts.T).T + pos


# ── Measurement Simulation ────────────────────────────────────────────────────
def get_measurements(pos,heading):

    dists = np.linalg.norm(MAP_CONES-pos,axis=1)

    visible_idx = np.where(dists < SENSOR_RANGE)[0]
    visible = MAP_CONES[visible_idx]

    if len(visible)==0:
        return np.zeros((0,2)),np.array([])

    c,s = np.cos(heading),np.sin(heading)
    R = np.array([[c,s],[-s,c]])

    local = (R @ (visible-pos).T).T

    noisy = local + np.random.normal(0,NOISE_STD,local.shape)

    return noisy,visible_idx


def step_kinematic(pos,heading,velocity,steering):

    new_pos = pos.copy()

    new_pos[0] += velocity*np.cos(heading)*DT
    new_pos[1] += velocity*np.sin(heading)*DT

    new_heading = angle_wrap(
        heading + (velocity/WHEELBASE)*np.tan(steering)*DT
    )

    return new_pos,new_heading


# ── Drawing Functions ─────────────────────────────────────────────────────────
def draw_track(ax):

    ax.scatter(BLUE_CONES[:,0],BLUE_CONES[:,1],c="royalblue",marker="^",s=65,alpha=0.4)
    ax.scatter(YELLOW_CONES[:,0],YELLOW_CONES[:,1],c="gold",marker="^",s=65,alpha=0.4)
    ax.scatter(BIG_ORANGE[:,0],BIG_ORANGE[:,1],c="darkorange",marker="s",s=100)


def draw_car(ax,pos,heading):

    ax.scatter(pos[0],pos[1],c="red",s=160)

    ax.arrow(pos[0],pos[1],
             2.2*np.cos(heading),
             2.2*np.sin(heading),
             head_width=0.8,
             fc="red",ec="red")


def setup_ax(ax,subtitle=""):

    ax.set_xlim(-28,28)
    ax.set_ylim(-22,22)
    ax.set_aspect("equal")
    ax.grid(True,alpha=0.25)

    if subtitle:
        ax.set_title(subtitle)


# ── Bot Base ─────────────────────────────────────────────────────────────────
class Bot:

    def __init__(self):

        self.pos = CAR_START_POS.copy()
        self.heading = CAR_START_HEADING


# ── Solution ─────────────────────────────────────────────────────────────────
class Solution(Bot):

    def __init__(self):

        super().__init__()

        self._global_meas = np.zeros((0,2))
        self._assoc = np.array([],dtype=int)

        self.exact_history = []
        self.near_history = []

    def data_association(self, measurements, current_map):

        if len(measurements) == 0 or len(current_map) == 0:
            self._assoc = np.array([], dtype=int)
            self._global_meas = np.zeros((0,2))
            # Count this frame: every measurement is a missed association
            n_meas = len(measurements)
            if n_meas > 0:
                self.exact_history.append(0.0)
                self.near_history.append(0.0)
            return

        # convert measurements to global frame
        gm = local_to_global(measurements, self.pos, self.heading)
        self._global_meas = gm

        # compute distance matrix
        D = distance.cdist(gm, current_map)

       
        noise_radius = 3.0 * NOISE_STD                 # pure sensor noise term
        motion_slack = 0.25 * SPEED * DT               # ~25% of one-frame travel
        GATE = noise_radius + motion_slack

        # apply gating — entries beyond GATE become very large so Hungarian
        # treats them as infeasible without changing the optimal structure
        gated_D = D.copy()
        gated_D[gated_D > GATE] = 1e6

        # Hungarian assignment
        rows, cols = linear_sum_assignment(gated_D)

        assoc = -np.ones(len(gm), dtype=int)

        for r, c in zip(rows, cols):

            if D[r, c] <= GATE:
                assoc[r] = c

        self._assoc = assoc

        # ── Temporal consistency ───────────────────────────────────────────────
        # Lazy-init so we stay within data_association only
        if not hasattr(self, '_hit_count'):
            self._hit_count    = {}
            self.CONFIRM_THRESH = 3
       

        matched_this_frame = set(c for c in assoc if c != -1)

        # decay counters for landmarks not seen this frame
        for idx in list(self._hit_count.keys()):
            if idx not in matched_this_frame:
                self._hit_count[idx] = 0

        # update counters for landmarks matched this frame
        for c in matched_this_frame:
            self._hit_count[c] = self._hit_count.get(c, 0) + 1

        confirmed_assoc = assoc.copy()
        for i, c in enumerate(assoc):
            if c != -1 and self._hit_count.get(c, 0) < self.CONFIRM_THRESH:
                confirmed_assoc[i] = -1

        self._assoc = confirmed_assoc



# ── Visualization ─────────────────────────────────────────────────────────────
def make_problem1():

    sol = Solution()

    fig,ax = plt.subplots(figsize=(10,7))

    def update(frame):

        ax.clear()

        steer = pure_pursuit(sol.pos,sol.heading,CENTERLINE)

        meas,_ = get_measurements(sol.pos,sol.heading)

        sol.data_association(meas,MAP_CONES)

        sol.pos,sol.heading = step_kinematic(sol.pos,sol.heading,SPEED,steer)

        draw_track(ax)

        if len(sol._global_meas)>0:

            for idx,gm in zip(sol._assoc,sol._global_meas):

                if idx == -1:
                    continue

                mc = MAP_CONES[idx]

                ax.plot([gm[0],mc[0]],[gm[1],mc[1]],"lime",lw=2)

            ax.scatter(sol._global_meas[:,0],
                       sol._global_meas[:,1],
                       c="cyan",s=40,zorder=5)

        draw_car(ax,sol.pos,sol.heading)

        exact    = np.mean(sol.exact_history)    if sol.exact_history    else 0
        near     = np.mean(sol.near_history)     if sol.near_history     else 0
        coverage = np.mean(sol.coverage_history) if hasattr(sol, 'coverage_history') and sol.coverage_history else 0

        setup_ax(ax,
                 f"Frame {frame+1}/{N_FRAMES} | Precision Exact: {exact*100:.1f}% | Near: {near*100:.1f}% | Coverage: {coverage*100:.1f}%")

    ani = FuncAnimation(fig,update,frames=N_FRAMES,interval=100)

    return fig,ani


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":

    fig,ani = make_problem1()

    plt.show()
