import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple


# ----------------------------
# Inputs & dataclass
# ----------------------------
@dataclass
class Pipe:
    D: float        # pipe ID [m]
    theta: float    # inclination [rad] (0 = horizontal, +upflow)
    eps: float = 0  # roughness [m] (optional)


@dataclass
class Fluids:
    rho_l: float    # liquid density [kg/m3]
    mu_l: float     # liquid viscosity [Pa·s]
    rho_g: float    # gas density [kg/m3]
    mu_g: float     # gas viscosity [Pa·s]
    sigma: float    # surface tension [N/m]


# ----------------------------
# Utilities (dimensionless)
# ----------------------------
g = 9.80665


def Re(rho, v, D, mu):
    return rho * v * D / mu


def Fr(v, D):
    return v / np.sqrt(g * D)


def We(rho, v, D, sigma):
    return rho * v**2 * D / sigma


# ----------------------------
# Strategy-style classifier interface
# ----------------------------
class FlowRegimeModel:
    name = "base"
    def classify(self, jl: float, jg: float, pipe: Pipe, fluids: Fluids) -> str:
        raise NotImplementedError


# ----------------------------
# 2.1 Mandhane-like demo classifier (educational)
# NOTE: This is a *rough* pedagogical approximation to reproduce a map feel.
# For serious work, replace with actual Mandhane curves or a mechanistic model.
# ----------------------------
class MandhaneDemo(FlowRegimeModel):
    name = "Mandhane-demo"

    def classify(self, jl: float, jg: float, pipe: Pipe, fluids: Fluids) -> str:
        # Use simple heuristics in (jl, jg):
        # (Log-logic inspired regions; tune thresholds to literature if you want closer match)
        jl_log = np.log10(max(jl, 1e-6))
        jg_log = np.log10(max(jg, 1e-6))

        # Very low gas, modest liquid -> bubble/stratified
        if jg_log < -2.0 and jl_log > -2.5:
            return "bubble"

        # Low gas & low liquid -> stratified smooth
        if jg_log < -1.5 and jl_log < -2.0:
            return "stratified"

        # Moderate gas with low liquid -> wavy/stratified-wavy
        if -1.5 <= jg_log <= -0.3 and jl_log < -1.0:
            return "wavy"

        # Mid gas & mid liquid -> intermittent/slug
        if -1.3 <= jg_log <= 0.5 and -2.0 <= jl_log <= 0.3:
            return "slug/intermittent"

        # High gas, low liquid -> annular/mist
        if jg_log > 0.3 and jl_log < -0.5:
            return "annular"

        # High both -> dispersed (frothy) or churn-ish
        if jg_log > -0.3 and jl_log > -0.3:
            return "dispersed"

        # Fallback
        return "transition"


# ----------------------------
# 2.2 Taitel–Dukler placeholder (fill with real criteria)
# ----------------------------
class TaitelDukler(FlowRegimeModel):
    name = "Taitel–Dukler-1976"

    def classify(self, jl: float, jg: float, pipe: Pipe, fluids: Fluids) -> str:
        D, th = pipe.D, pipe.theta
        rho_l, mu_l = fluids.rho_l, fluids.mu_l
        rho_g, mu_g, sigma = fluids.rho_g, fluids.mu_g, fluids.sigma

        # Example: compute superficial Re, Fr, We
        Re_l = Re(rho_l, jl, D, mu_l)
        Re_g = Re(rho_g, jg, D, mu_g)
        Fr_l = Fr(jl, D)
        Fr_g = Fr(jg, D)
        We_l = We(rho_l, jl, D, sigma)
        We_g = We(rho_g, jg, D, sigma)

        # TODO: implement KH onset for stratified->intermittent,
        # flooding/envelopes for annular, dispersed/bubble boundaries, and inclination corrections.
        # Return one of your regimes based on those criteria.
        # For now, just route to a simple proxy:
        return MandhaneDemo().classify(jl, jg, pipe, fluids)


# ----------------------------
# 2.3 Beggs & Brill placeholder (fill with real criteria)
# ----------------------------
class BeggsBrill(FlowRegimeModel):
    name = "Beggs-Brill-1973"

    def classify(self, jl: float, jg: float, pipe: Pipe, fluids: Fluids) -> str:
        # TODO:
        # 1) Compute mixture properties & inclination correction.
        # 2) Evaluate the velocity numbers used by Beggs & Brill
        # 3) Classify into segregated / intermittent / distributed (or transition).
        return MandhaneDemo().classify(jl, jg, pipe, fluids)


# ----------------------------
# 3) Map builder
# ----------------------------
def build_map(model: FlowRegimeModel,
              pipe: Pipe,
              fluids: Fluids,
              jl_range: Tuple[float, float]=(1e-3, 5.0),
              jg_range: Tuple[float, float]=(1e-3, 20.0),
              n: int=250):

    jl_vals = np.logspace(np.log10(jl_range[0]), np.log10(jl_range[1]), n)
    jg_vals = np.logspace(np.log10(jg_range[0]), np.log10(jg_range[1]), n)

    JL, JG = np.meshgrid(jl_vals, jg_vals)
    regimes = np.empty(JL.shape, dtype=object)

    # Classify each grid point
    for i in range(JL.shape[0]):
        for j in range(JL.shape[1]):
            regimes[i,j] = model.classify(JL[i,j], JG[i,j], pipe, fluids)

    return jl_vals, jg_vals, regimes


# ----------------------------
# 4) Plot
# ----------------------------
def plot_map(jl_vals, jg_vals, regimes, title="Flow Pattern Map"):
    # Assign a color per regime
    palette = {
        "stratified":       0,
        "wavy":             1,
        "slug/intermittent":2,
        "annular":          3,
        "bubble":           4,
        "dispersed":        5,
        "transition":       6
    }
    Z = np.vectorize(lambda r: palette.get(r, 6))(regimes)

    plt.figure(figsize=(7.5, 6.5))
    plt.pcolormesh(jl_vals, jg_vals, Z, shading='nearest')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r"Liquid superficial velocity $j_L$ [m/s]")
    plt.ylabel(r"Gas superficial velocity $j_G$ [m/s]")
    plt.title(title)

    # Legend
    inv_palette = {v:k for k,v in palette.items()}
    patches = [plt.Line2D([0],[0], marker='s', linestyle='', label=inv_palette[i]) for i in sorted(inv_palette)]
    plt.legend(handles=patches, title="Regime", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 5) Example run
# ----------------------------
if __name__ == "__main__":
    pipe = Pipe(D=0.05, theta=0.0)  # 2-inch horizontal
    fluids = Fluids(
        rho_l=998.0,  mu_l=1.0e-3,   # water @ ~20°C
        rho_g=1.2,    mu_g=1.8e-5,   # air
        sigma=0.072
    )

    model = MandhaneDemo()          # swap to TaitelDukler() or BeggsBrill() once implemented
    jl, jg, R = build_map(model, pipe, fluids,
                          jl_range=(1e-3, 5.0),
                          jg_range=(1e-3, 20.0),
                          n=220)
    plot_map(jl, jg, R, title=f"{model.name} — D={pipe.D*1000:.0f} mm, θ={np.degrees(pipe.theta):.0f}°")
