"""Robot-arm demo: the Bat Algorithm solves inverse kinematics for a 5-joint arm.
Left-click = move the target, right-click = place an obstacle. Every bat is one arm pose (5 joint angles).
Runs YOUR bat_algorithm (variant 'book'); needs the trace=None hook."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button

from algo import bat_algorithm_steps

LINKS = np.array([2.0, 1.6, 1.2, 0.9, 0.6])  # link lengths, total reach 6.3
N_BATS, T_MAX, SIGMA, ALPHA = 20, 50, 0.1, 0.97  # book defaults except n and t_max
OBST_R = 0.8
STEP_PER_TICK = 3
ANIMATION_STEPS = 30


def joints(theta):
    """(joint positions) for joint angles theta (each angle is relative to the previous link)."""
    a = np.cumsum(theta)
    pts = np.zeros((len(theta) + 1, 2))
    pts[1:, 0] = np.cumsum(LINKS * np.cos(a))
    pts[1:, 1] = np.cumsum(LINKS * np.sin(a))
    return pts


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def make_objective(target, obst):
    """squared distance of the hand to the target + a penalty when any link enters the obstacle."""

    def f(theta):
        pts = joints(theta)
        cost = np.sum((pts[-1] - target) ** 2)
        if obst is not None:
            for k in range(len(pts) - 1):
                d = seg_dist(obst, pts[k], pts[k + 1])
                if d < OBST_R:
                    cost += 10 * (OBST_R - d) ** 2
        return cost

    return f


def bat_arm_gui():
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_axes([0.04, 0.16, 0.50, 0.78])
    ax_c = fig.add_axes([0.62, 0.50, 0.35, 0.38])
    ax_t = fig.add_axes([0.62, 0.16, 0.35, 0.25])
    ax_t.axis("off")
    it_s = Slider(
        fig.add_axes([0.10, 0.07, 0.30, 0.03]),
        "iteration",
        0,
        T_MAX,
        valinit=0,
        valstep=1,
    )
    play_b = Button(fig.add_axes([0.43, 0.065, 0.07, 0.04]), "Pause")
    rand_b = Button(fig.add_axes([0.62, 0.065, 0.14, 0.04]), "Random target")
    clr_b = Button(fig.add_axes([0.78, 0.065, 0.14, 0.04]), "Clear obstacle")

    st = dict(
        target=np.array([3.5, 3.0]),
        obst=None,
        trace=[],
        runner=None,
        
        # Animation state
        display_best=None,
        animation_from=None,
        animation_to=None,
        animation_step=0,
        
        playing=True,
        art={},
        runs=0,
    )

    def build():
        ax.clear()
        R = LINKS.sum()
        ax.set_xlim(-R - 0.3, R + 0.3)
        ax.set_ylim(-R - 0.3, R + 0.3)
        ax.set_aspect("equal")
        ax.add_patch(Circle((0, 0), R, fill=False, ls=":", color="gray"))
        ax.plot(0, 0, "ks", ms=9)
        if st["obst"] is not None:
            ax.add_patch(Circle(st["obst"], OBST_R, color="tab:red", alpha=0.35))
        ax.plot(*st["target"], "r*", ms=18, zorder=5)
        ax.set_title("left-click: target   |   right-click: obstacle", fontsize=10)
        art = st["art"]
        art["ghosts"] = [
            ax.plot([], [], "-", color="tab:blue", alpha=0.18, lw=1)[0]
            for _ in range(N_BATS)
        ]
        (art["best"],) = ax.plot([], [], "o-", color="k", lw=3, ms=6, zorder=4)
        (art["hand"],) = ax.plot([], [], "o", color="m", ms=10, zorder=6)
        ax_c.clear()

        # Create the convergence line.
        (art["convergence"],) = ax_c.plot(
            [], [],
            color="tab:blue",
            lw=2,
            marker=".",
            markersize=4,
        )

        # Create the red marker indicating the current iteration.
        (art["marker"],) = ax_c.plot([], [], "ro", zorder=5)

        # Use logarithmic scale to visualize large differences in fitness.
        ax_c.set_yscale("log")

        ax_c.set_xlim(0, T_MAX)

        ax_c.set_xlabel("iteration")
        ax_c.set_ylabel("f_best (cost)")
        ax_c.set_title("convergence")
        ax_t.clear()
        ax_t.axis("off")
        art["text"] = ax_t.text(0, 1, "", va="top", family="monospace", fontsize=10)

    def draw(t, pose=None):
        if not st["trace"]:
            return

        # Only display iterations that have already been calculated
        t = int(min(max(t, 0), len(st["trace"]) - 1))
        fr = st["trace"][t]
        best_pose = fr["best"] if pose is None else pose
        art = st["art"]
        for g, th in zip(art["ghosts"], fr["pos"]):
            p = joints(th)
            g.set_data(p[:, 0], p[:, 1])
        pb = joints(best_pose)
        art["best"].set_data(pb[:, 0], pb[:, 1])
        art["hand"].set_data([pb[-1, 0]], [pb[-1, 1]])
        # Extract the best fitness values calculated so far.
        iterations = np.arange(len(st["trace"]))

        costs = np.maximum(
            [frame["f"] for frame in st["trace"]],
            1e-300,
        )

        # Update the convergence curve.
        art["convergence"].set_data(iterations, costs)

        # Update the red marker to indicate the current iteration.
        art["marker"].set_data([t], [max(fr["f"], 1e-300)])

        # Automatically adjust the Y-axis to fit the updated curve.
        ax_c.relim()
        ax_c.autoscale_view(scalex=False, scaley=True)
        miss = np.linalg.norm(pb[-1] - st["target"])
        art["text"].set_text(
            f"iteration     : {t}\nhand-target gap: {miss:.4f}\ncost (f_best)  : {fr['f']:.3e}\n"
            f"bats (arm poses): {N_BATS}"
        )
        fig.canvas.draw_idle()

    def solve(restart_play=True):
        # Start from the arm's currently displayed pose.
        previous_best = None

        if st["display_best"] is not None:
            previous_best = st["display_best"].copy()
        elif st["trace"]:
            previous_best = st["trace"][-1]["best"].copy()

        # Stop the previous optimization and animation.
        st["runner"] = None
        st["animation_to"] = None
        st["animation_step"] = 0

        f = make_objective(st["target"], st["obst"])

        st["runs"] += 1

        st["runner"] = bat_algorithm_steps(
            f,
            dim=len(LINKS),
            lb=-np.pi,
            ub=np.pi,
            n=N_BATS,
            t_max=T_MAX,
            alpha=ALPHA,
            sigma=SIGMA,
            seed=st["runs"],
        )

        initial_frame = next(st["runner"])

        st["trace"] = [initial_frame]

        st["display_best"] = initial_frame["best"].copy()
        st["animation_from"] = st["display_best"].copy()
        st["animation_to"] = None
        st["animation_step"] = 0

        build()

        it_s.eventson = False
        it_s.set_val(0)
        it_s.eventson = True

        draw(0, pose=st["display_best"])

        if restart_play and not st["playing"]:
            toggle()

    def toggle(_=None):
        st["playing"] = not st["playing"]
        play_b.label.set_text("Pause" if st["playing"] else "Play")
        fig.canvas.draw_idle()

    def tick():
        if not st["playing"] or st["runner"] is None:
            return

        # -------------------------------------------------
        # PART 1: Animate between the previous and new pose
        # -------------------------------------------------
        if st["animation_to"] is not None:

            st["animation_step"] += 1

            progress = st["animation_step"] / ANIMATION_STEPS

            start = st["animation_from"]
            end = st["animation_to"]

            # Calculate the shortest angular movement.
            delta = (end - start + np.pi) % (2 * np.pi) - np.pi

            # Interpolate the joint angles.
            st["display_best"] = start + progress * delta

            # Finish the animation exactly at the new pose.
            if st["animation_step"] >= ANIMATION_STEPS:
                st["display_best"] = end.copy()

                st["animation_from"] = end.copy()
                st["animation_to"] = None
                st["animation_step"] = 0

            draw(len(st["trace"]) - 1, pose=st["display_best"])

            return

        # -------------------------------------------------
        # PART 2: Calculate the next Bat Algorithm iteration
        # -------------------------------------------------
        try:
            frame = next(st["runner"])

        except StopIteration:
            st["runner"] = None

            if st["playing"]:
                toggle()

            return

        st["trace"].append(frame)

        # Prepare the next animation.
        st["animation_from"] = st["display_best"].copy()
        st["animation_to"] = frame["best"].copy()
        st["animation_step"] = 0

        # Update the iteration slider.
        it_s.eventson = False
        it_s.set_val(frame["iteration"])
        it_s.eventson = True

        # Update the other visual elements while keeping
        # the arm at its previous position.
        draw(frame["iteration"], pose=st["display_best"])

        if frame["iteration"] >= T_MAX:
            # The final pose still needs to be animated.
            st["runner"] = None

    def on_click(ev):
        tb = getattr(fig.canvas, "toolbar", None)
        if ev.inaxes is not ax or (tb is not None and getattr(tb, "mode", "") != ""):
            return
        p = np.array([ev.xdata, ev.ydata])
        if ev.button == 1:
            st["target"] = p
        elif ev.button == 3:
            st["obst"] = p
        else:
            return
        solve()

    def random_target(_=None):
        r = np.random.uniform(1.5, 6.0)
        a = np.random.uniform(0, 2 * np.pi)
        st["target"] = r * np.array([np.cos(a), np.sin(a)])
        solve()

    def clear_obst(_=None):
        st["obst"] = None
        solve()

    it_s.on_changed(draw)
    play_b.on_clicked(toggle)
    rand_b.on_clicked(random_target)
    clr_b.on_clicked(clear_obst)
    fig.canvas.mpl_connect("button_press_event", on_click)
    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(tick)
    timer.start()

    solve(restart_play=False)
    fig._keepalive = (it_s, play_b, rand_b, clr_b, timer)
    return fig, dict(
        it_s=it_s, st=st, solve=solve, tick=tick, click=on_click, toggle=toggle
    )


def show_bat_arm_gui():
    bat_arm_gui()
    plt.show()
