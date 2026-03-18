"""
This module collects a few little helper functions to support
the notebooks for the Berta-Thompson et al. (2026) cosmic shoreline paper.
"""

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
    "relative_instellation",
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

    for k in [
        "relative_escape_velocity",
        "relative_instellation",
        "stellar_luminosity",
    ]:
        ok = np.isfinite(table[f"{k}"])
        ok *= np.isfinite(table[f"{k}_uncertainty"])
        table = table[ok]
    return table


def save_organized_populations(A, directory="organized-exoatlas-populations"):
    mkdir(directory)
    for fluxlimit in A:
        for label in A[fluxlimit]:
            for subpop in A[fluxlimit][label]:
                A[fluxlimit][label][subpop].save(
                    f"{directory}/{fluxlimit}+{label}+{subpop}.ecsv"
                )


def load_organized_populations(
    directory="organized-exoatlas-populations",
    subset="*",
    fluxlimit="*",
    label="*",
    subpop="*",
):
    A = {}
    files = glob.glob(f"{directory}/{subset}+{fluxlimit}+{label}+{subpop}.ecsv")
    for f in tqdm(files):
        subset, atmosphere_fluxlimit, label, subpop = (
            os.path.basename(f).split(".ecsv")[0].split("+")
        )
        fluxlimit = f"{subset}+{atmosphere_fluxlimit}"
        if fluxlimit not in A:
            A[fluxlimit] = {}
        if label not in A[fluxlimit]:
            A[fluxlimit][label] = {}
        A[fluxlimit][label][subpop] = Population(f)
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


def make_safe_for_latex(s):
    s = clean(s).replace("0", "o").replace("CO2", "COO")
    for k, v in dict(
        one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, zero=0
    ).items():
        s = s.replace(str(v), k)
    return s


def latexify_arviz_posterior(posterior, label=""):

    func_dict = {
        "median": np.median,
        "lower": lambda x: np.median(x) - np.percentile(x, 50 - 68.3 / 2),
        "upper": lambda x: np.percentile(x, 50 + 68.3 / 2) - np.median(x),
    }

    summary = az.summary(posterior, stat_funcs=func_dict)
    lines = []
    for k in summary["median"].keys():

        s = latexify_confidence_interval(
            median=summary["median"][k],
            lower=summary["lower"][k],
            upper=summary["upper"][k],
        )

        lines.append(
            rf"\newcommand{{\{make_safe_for_latex(label)}{make_safe_for_latex(k)}}}{{{s}}}"
            + "\n"
        )
        s = f'{summary["median"][k]:.3g}'
        lines.append(
            rf"\newcommand{{\{make_safe_for_latex(label)}{make_safe_for_latex(k)+'justvalue'}}}{{{s}}}"
            + "\n"
        )
    lines.append("\n")
    return lines


def best_and_sampled(posterior):
    summary = az.summary(posterior, fluxlimit="all", stat_focus="median")
    best_parameters = summary["median"]

    df = posterior.to_dataframe(var_names=["log_f_0", "p", "q", "ln_w"])
    N_samples = 100
    sampled_parameters = df[:: int(len(df) / N_samples)]
    return best_parameters, sampled_parameters


class ShorelineStepAlreadyDoneError(Exception):
    """Base custom error for shoreline processing."""

    pass
