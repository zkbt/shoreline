from exoatlas import *
from exoatlas.visualizations import *
import arviz as az
import jax

try:
    jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError:
    pass
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import arviz as az
import numpyro
import corner


# define some table columns
basic_columns = ["name", "is_exoplanet", "has_atmosphere"]
predictor_columns = [
    "mass",
    "radius",
    "relative_escape_velocity",
    "relative_insolation",
    "stellar_luminosity",
    "stellar_mass",
]
uncertainty_columns = [f"{k}_uncertainty" for k in predictor_columns]

columns = basic_columns + predictor_columns + uncertainty_columns


def convert_labeled_populations_into_table(labeled, skip_minor=True):

    subsets = []
    for c in ["yes", "no"]:
        for k in labeled[c]:
            if skip_minor:
                if k == "minor":
                    continue
            t = labeled[c][k].create_table(columns, kludge=True)
            subsets.append(t)
    table = vstack(subsets)

    for k in ["relative_escape_velocity", "relative_insolation", "stellar_luminosity"]:
        ok = np.isfinite(table[f"{k}"])
        ok *= np.isfinite(table[f"{k}_uncertainty"])
        table = table[ok]
    return table


def save_organized_populations(A, directory="organized-exoatlas-populations"):
    mkdir(directory)
    for kind in A:
        for label in A[kind]:
            for subpop in A[kind][label]:
                A[kind][label][subpop].save(f"{directory}/{kind}-{label}-{subpop}.ecsv")


def load_organized_populations(
    directory="organized-exoatlas-populations",
    subset="*",
    kind="*",
    label="*",
    subpop="*",
):
    A = {}
    files = glob.glob(f"{directory}/{subset}-{kind}-{label}-{subpop}.ecsv")
    for f in tqdm(files):
        subset, atmosphere_kind, label, subpop = (
            os.path.basename(f).split(".ecsv")[0].split("-")
        )
        kind = f"{subset}-{atmosphere_kind}"
        if kind not in A:
            A[kind] = {}
        if label not in A[kind]:
            A[kind][label] = {}
        A[kind][label][subpop] = Population(f)
    return A


def visualize_labeled_populations(labeled):
    g = GridGallery(
        rows=[Flux(lim=[1e-4 / 5, 5e4]), Radius(lim=[0.001, 30])],
        cols=[
            RelativeEscapeVelocity(lim=[0.0001, 100]),
            Mass(lim=[1e-11, 3000]),
            Radius(lim=[0.001, 30]),
            StellarLuminosity(lim=[0.0001, 100]),
        ],
        map_type=ErrorMap,
        mapsize=(3, 3),
    )
    for k in labeled:
        g.build(labeled[k])


def log_f_shoreline(log_f_0=0.0, p=0.0, q=0.0, log_v=0, log_L=0):
    return log_f_0 + p * log_v + q * log_L


def probability_of_atmosphere(
    log_f_0=1.0, p=4.0, q=0.0, ln_w=0, log_v=0, log_L=0, log_f=0
):
    distance_from_shoreline = log_f - log_f_shoreline(
        log_f_0=log_f_0, p=p, q=q, log_v=log_v, log_L=log_L
    )
    width_of_shoreline = jnp.exp(ln_w)
    return 1 / (1 + jnp.exp(distance_from_shoreline / width_of_shoreline))


import jax.numpy as jnp


def plot_shoreline_3D(pops=None, log_f_0=1.0, p=4.0, q=1.0, ln_w=0.0, **kw):
    """ """

    for kind in pops:
        for k in pops[kind]:
            pops[kind][k].annotate_planets = len(pops[kind][k]) < 20
    # set up 2D grid for background colormap
    log_v_1d = jnp.linspace(-4, 2, 1000)
    log_f_1d = jnp.linspace(-5, 5, 1000)
    log_v_2d, log_f_2d = jnp.meshgrid(log_v_1d, log_f_1d)

    N = 4
    log_centers = np.linspace(0, -3, N)
    log_widths = np.abs(np.gradient(log_centers))
    lowers = 10 ** (log_centers - log_widths / 2) * u.Lsun
    uppers = 10 ** (log_centers + log_widths / 2) * u.Lsun
    centers = 10**log_centers * u.Lsun

    fi = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = plt.GridSpec(3, N, height_ratios=[1, 1, 1], figure=fi)

    ax_stellar_radius = plt.subplot(gs[0, :])
    plt.sca(ax_stellar_radius)
    m = ErrorMap(
        xaxis=StellarLuminosity(lim=[max(uppers), min(lowers)]),
        yaxis=Radius(lim=[0.1, 30]),
        ax=ax_stellar_radius,
    )
    for k in ["yes", "no"]:  # pops:
        m.build(pops[k])
    ax_stellar_radius.xaxis.set_ticks_position("top")
    ax_stellar_radius.xaxis.set_label_position("top")

    ax_stellar_flux = plt.subplot(gs[1, :])
    plt.sca(ax_stellar_flux)
    m = ErrorMap(
        xaxis=StellarLuminosity(lim=[max(uppers), min(lowers)]),
        yaxis=Flux(lim=[0.2e-4, 5e4]),
        ax=ax_stellar_flux,
    )
    for k in ["yes", "no"]:  # pops:
        m.build(pops[k])
    # ax_stellar_flux.xaxis.set_ticks_position("top")
    # ax_stellar_flux.xaxis.set_label_position("top")

    ax_shoreline = None
    for i in range(N):
        ax_shoreline = plt.subplot(gs[-1, i], sharex=ax_shoreline, sharey=ax_shoreline)
        m = ErrorMap(
            xaxis=RelativeEscapeVelocity(lim=[1e-4, 100], kludge=True),
            yaxis=Flux(lim=[0.2e-4, 5e4]),
            ax=ax_shoreline,
            size=100,
        )

        for c in ["yes", "no"]:  # pops
            for k in pops[c]:
                pop = pops[c][k]
                allowed = (pop.stellar_luminosity() > lowers[i]) & (
                    pop.stellar_luminosity() <= uppers[i]
                )
                if sum(allowed) > 0:
                    subset = pop[allowed]
                    subset._plotkw = pop._plotkw
                    subset.label = pop.label
                    if len(subset) < 10:
                        subset.annotate_planets = True
                    dots = subset[:]
                    dots.bubble_anyway = True
                    dots.s = 64
                    # m.build(dots)
                    m.build(subset)

        log_P_2d = probability_of_atmosphere(
            log_v=log_v_2d,
            log_f=log_f_2d,
            log_L=log_centers[i],
            p=p,
            q=q,
            log_f_0=log_f_0,
            ln_w=ln_w,
        )
        background = plt.pcolormesh(
            10**log_v_2d,
            10**log_f_2d,
            log_P_2d,
            cmap=one2another("burlywood", "lightskyblue"),
            alpha=1,
            zorder=-1e9,
            rasterized=True,
        )
        shoreline_kw = dict(color="black", alpha=0.3)
        shoreline_solar = plt.plot(
            10**log_v_1d,
            10 ** log_f_shoreline(log_f_0=log_f_0, p=p, q=q, log_v=log_v_1d, log_L=0),
            linestyle="--",
            **shoreline_kw,
        )
        shoreline_this = plt.plot(
            10**log_v_1d,
            10
            ** log_f_shoreline(
                log_f_0=log_f_0, p=p, q=q, log_v=log_v_1d, log_L=log_centers[i]
            ),
            linestyle=":",
            **shoreline_kw,
        )

        # titled = plt.title(f'{log_P=:.2f}\n{log_f_0=:.2f}, {p=:.2f}, {q=:.2f}, {ln_w=:.2f}' )
        # plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")

        plt.title(f'{centers[i].to_string(format="latex_inline")}')
        if i > 0:
            plt.setp(ax_shoreline.get_yticklabels(), visible=False)
            plt.ylabel("")
