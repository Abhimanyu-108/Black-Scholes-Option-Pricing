# %% [markdown]
# # Black–Scholes Option Pricing, Monte Carlo Validation & a VC Application
#
# **Author:** Abhimanyu Gaur · B.Sc. Statistics (Hons.), Ramjas College, University of Delhi
#
# **What this notebook does**
# 1. Prices European options analytically (Black–Scholes–Merton, with continuous dividend yield)
# 2. Validates the closed form with Monte Carlo, including **antithetic variance reduction** and standard errors
# 3. Computes all Greeks analytically and **cross-checks them against finite differences**
# 4. Recovers implied volatility (Brent root-finding)
# 5. Applies the model to a venture-style problem: **valuing a startup ESOP / warrant grant**
#
# > Educational project. Not investment advice.

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
SEED = 42  # reproducibility

# Base-case parameters
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
print(f"S={S}, K={K}, T={T}, r={r:.0%}, sigma={sigma:.0%}, q={q:.0%}")

# %% [markdown]
# ## 2. Model assumptions
# - Underlying follows Geometric Brownian Motion; constant volatility and risk-free rate
# - No arbitrage, frictionless markets, European exercise
# - Continuous dividend yield `q` (set `q=0` for the classic Black–Scholes case)
#
# Real markets violate several of these (volatility smile, jumps, early exercise), which is why Black–Scholes
# is best used as a **benchmark and quoting convention** rather than as ground truth.

# %% [markdown]
# ## 3. Analytical pricing

# %%
def d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Black–Scholes–Merton price of a European call or put."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


call = bs_price(S, K, T, r, sigma, q, "call")
put = bs_price(S, K, T, r, sigma, q, "put")
print(f"Call = {call:.4f} | Put = {put:.4f}")

# Put–call parity check: C - P = S*e^{-qT} - K*e^{-rT}
lhs = call - put
rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
print(f"Put–call parity: {lhs:.6f} vs {rhs:.6f}  (diff = {abs(lhs - rhs):.2e})")

# %%
strikes = np.linspace(60, 140, 100)
fig, ax = plt.subplots()
ax.plot(strikes, bs_price(S, strikes, T, r, sigma, q, "call"), label="Call")
ax.plot(strikes, bs_price(S, strikes, T, r, sigma, q, "put"), label="Put")
ax.axvline(S, ls="--", c="grey", lw=1, label="Spot")
ax.set(xlabel="Strike", ylabel="Option price", title="Option Price vs Strike")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Monte Carlo validation
# Under the risk-neutral measure: $S_T = S\,e^{(r-q-\sigma^2/2)T + \sigma\sqrt{T}Z}$, with $Z\sim N(0,1)$.
# The price is the discounted mean payoff. We report a **standard error** and compare plain MC with
# **antithetic variates** (pairing each draw `Z` with `-Z`) to cut variance at no extra cost.

# %%
def mc_price(S, K, T, r, sigma, q=0.0, option="call", n=100_000,
             antithetic=True, rng=None):
    """Monte Carlo price and standard error."""
    rng = rng or np.random.default_rng(SEED)
    if antithetic:
        half = n // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if option == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    if antithetic:  # average each (Z, -Z) pair first so the SE is valid
        disc = 0.5 * (disc[:half] + disc[half:])
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


rows = []
for opt in ("call", "put"):
    bs = bs_price(S, K, T, r, sigma, q, opt)
    for anti in (False, True):
        p, se = mc_price(S, K, T, r, sigma, q, opt, n=200_000, antithetic=anti)
        rows.append({"Option": opt, "Antithetic": anti, "Black–Scholes": bs,
                     "Monte Carlo": p, "Std error": se,
                     "Abs error": abs(p - bs),
                     "BS within 95% CI": abs(p - bs) <= 1.96 * se})
pd.DataFrame(rows).round(4)

# %%
# Convergence: mean absolute error over repeated runs, vs the theoretical 1/sqrt(n) rate
sizes = np.array([1_000, 5_000, 10_000, 50_000, 100_000, 500_000])
reps = 30
rng = np.random.default_rng(SEED)
err = {False: [], True: []}
for n in sizes:
    for anti in (False, True):
        e = [abs(mc_price(S, K, T, r, sigma, q, "call", n, anti, rng)[0] - call)
             for _ in range(reps)]
        err[anti].append(np.mean(e))

fig, ax = plt.subplots()
ax.loglog(sizes, err[False], "o-", label="Plain MC")
ax.loglog(sizes, err[True], "s-", label="Antithetic MC")
ref = err[False][0] * np.sqrt(sizes[0] / sizes)
ax.loglog(sizes, ref, "k--", lw=1, label=r"$1/\sqrt{n}$ reference")
ax.set(xlabel="Number of simulations", ylabel=f"Mean absolute error ({reps} runs)",
       title="Monte Carlo Convergence to Black–Scholes")
ax.legend()
plt.show()

# %% [markdown]
# ## 5. Greeks
# Analytical Greeks for calls and puts. Conventions: **vega is per 1 volatility point**, **theta is per calendar day**,
# **rho is per 1 percentage point** of rate — the way traders quote them.

# %%
def greeks(S, K, T, r, sigma, q=0.0, option="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf, sq = norm.pdf(d1), np.sqrt(T)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    gamma = eq * pdf / (S * sigma * sq)
    vega = S * eq * pdf * sq
    if option == "call":
        delta = eq * norm.cdf(d1)
        theta = (-S * eq * pdf * sigma / (2 * sq)
                 - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1))
        rho = K * T * er * norm.cdf(d2)
    else:
        delta = eq * (norm.cdf(d1) - 1)
        theta = (-S * eq * pdf * sigma / (2 * sq)
                 + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1))
        rho = -K * T * er * norm.cdf(-d2)
    return {"Delta": delta, "Gamma": gamma,
            "Vega (per 1 vol pt)": vega / 100,
            "Theta (per day)": theta / 365,
            "Rho (per 1% rate)": rho / 100}


greek_table = pd.DataFrame({o: greeks(S, K, T, r, sigma, q, o) for o in ("call", "put")})
greek_table.round(4)

# %%
# Sanity check: analytical Greeks vs central finite differences of the price
def fd_greeks(S, K, T, r, sigma, q, option, h=1e-4):
    f = lambda **kw: bs_price(**{**dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option=option), **kw})
    return {
        "Delta": (f(S=S + h) - f(S=S - h)) / (2 * h),
        "Gamma": (f(S=S + h) - 2 * f() + f(S=S - h)) / h**2,
        "Vega (per 1 vol pt)": (f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h) / 100,
        "Theta (per day)": -(f(T=T + h) - f(T=T - h)) / (2 * h) / 365,
        "Rho (per 1% rate)": (f(r=r + h) - f(r=r - h)) / (2 * h) / 100,
    }


check = pd.DataFrame({"Analytical": greek_table["call"],
                      "Finite diff": pd.Series(fd_greeks(S, K, T, r, sigma, q, "call"))})
check["Abs diff"] = (check["Analytical"] - check["Finite diff"]).abs()
check.round(6)

# %%
S_grid = np.linspace(60, 140, 200)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for T_ in (0.1, 0.5, 1.0):
    axes[0].plot(S_grid, greeks(S_grid, K, T_, r, sigma, q, "call")["Delta"], label=f"T={T_}")
    axes[1].plot(S_grid, greeks(S_grid, K, T_, r, sigma, q, "call")["Gamma"], label=f"T={T_}")
    axes[2].plot(S_grid, greeks(S_grid, K, T_, r, sigma, q, "call")["Vega (per 1 vol pt)"], label=f"T={T_}")
for ax, t in zip(axes, ("Delta", "Gamma", "Vega (per 1 vol pt)")):
    ax.set(xlabel="Spot", title=f"Call {t}")
axes[0].legend()
plt.tight_layout()
plt.show()
# Takeaway: gamma and vega peak at-the-money and sharpen as expiry approaches.

# %%
# Call price surface: spot vs volatility
spots = np.linspace(70, 130, 60)
vols = np.linspace(0.05, 0.60, 60)
SS, VV = np.meshgrid(spots, vols)
fig, ax = plt.subplots()
im = ax.contourf(SS, VV * 100, bs_price(SS, K, T, r, VV, q, "call"), levels=20, cmap="viridis")
fig.colorbar(im, label="Call price")
ax.set(xlabel="Spot", ylabel="Volatility (%)", title="Call Price: Spot vs Volatility")
ax.grid(False)
plt.show()

# %% [markdown]
# ## 6. Implied volatility
# Invert Black–Scholes for σ given an observed price using Brent's method. A round-trip test confirms recovery.

# %%
def implied_vol(price, S, K, T, r, q=0.0, option="call"):
    f = lambda s: bs_price(S, K, T, r, s, q, option) - price
    try:
        return brentq(f, 1e-6, 5.0)
    except ValueError:  # price outside no-arbitrage bounds
        return np.nan


test_vols = [0.10, 0.20, 0.35, 0.60]
rt = pd.DataFrame({
    "True vol": test_vols,
    "Recovered vol": [implied_vol(bs_price(S, K, T, r, v, q, "call"), S, K, T, r, q) for v in test_vols],
})
rt["Abs error"] = (rt["True vol"] - rt["Recovered vol"]).abs()
rt

# %% [markdown]
# ## 7. VC application: valuing a startup ESOP / warrant grant
# Employee stock options and warrants are call options on the company's equity. A common approach for private
# companies is to price them with Black–Scholes, using:
# - **S**: fair value per share (e.g. from the latest round)
# - **K**: exercise price
# - **T**: expected option life
# - **σ**: volatility borrowed from **listed comparable companies** (private firms have no price history)
#
# The inputs below are **illustrative**, not from a real company.

# %%
S_esop, K_esop, T_esop, r_esop = 100.0, 120.0, 5.0, 0.07   # per share; 5y life; ~7% risk-free (illustrative)
vol_range = np.linspace(0.30, 1.00, 50)
values = bs_price(S_esop, K_esop, T_esop, r_esop, vol_range, 0.0, "call")

base_vol = 0.60
base_val = bs_price(S_esop, K_esop, T_esop, r_esop, base_vol, 0.0, "call")
print(f"Option value per share at {base_vol:.0%} vol: {base_val:.2f} "
      f"({base_val / S_esop:.1%} of share price)")

fig, ax = plt.subplots()
ax.plot(vol_range * 100, values)
ax.axvline(base_vol * 100, ls="--", c="grey", lw=1)
ax.set(xlabel="Assumed volatility (%)", ylabel="Option value per share",
       title="ESOP Value is Highly Sensitive to the Volatility Assumption")
plt.show()

# %%
# Sensitivity table: value per share across volatility and option life
grid = pd.DataFrame(
    {f"{t}y": [bs_price(S_esop, K_esop, t, r_esop, v, 0.0, "call") for v in (0.4, 0.6, 0.8)]
     for t in (3, 5, 7)},
    index=[f"vol {v:.0%}" for v in (0.4, 0.6, 0.8)])
grid.round(2)

# %% [markdown]
# **Reading the result.** Even a deep out-of-the-money grant (strike 20% above fair value) carries substantial
# value because of the long life and high volatility — and the value moves a lot with the volatility input.
# In practice analysts also apply a **discount for lack of marketability (DLOM)**, adjust for vesting and
# forfeiture, and may use a binomial/lattice or Monte Carlo model, since Black–Scholes ignores early exercise
# and non-traded underlyings.

# %% [markdown]
# ## 8. Summary
# - Closed-form Black–Scholes matches Monte Carlo within sampling error; antithetic variates reduce that error
# - Analytical Greeks agree with finite differences to numerical precision
# - Implied volatility is recovered accurately by Brent's method
# - The same framework prices employee options and warrants, with volatility as the key (and most uncertain) input
#
# **Limitations / next steps:** stochastic volatility (Heston), American options via binomial trees,
# a real implied-volatility surface from market data, and DLOM-adjusted ESOP valuation.
