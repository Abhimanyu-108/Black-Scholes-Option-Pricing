# Black–Scholes Option Pricing \& Model Validation

[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/black-scholes-option-pricing/blob/main/notebooks/Black_Scholes_Option_Pricing.ipynb)
!\[Python](https://img.shields.io/badge/python-3.9%2B-blue)
!\[License](https://img.shields.io/badge/license-MIT-green)

A compact quantitative-finance project that prices European options with the Black–Scholes–Merton model, **validates it independently** via Monte Carlo and finite differences, and applies it to a **venture-style problem: valuing startup ESOP / warrant grants**.

## What's inside

|Section|What it shows|
|-|-|
|Analytical pricing|Calls and puts with continuous dividend yield; put–call parity check|
|Monte Carlo validation|Standard errors, antithetic variance reduction, convergence vs the 1/√n rate|
|Greeks|Delta, gamma, vega, theta, rho for calls and puts, cross-checked against finite differences|
|Implied volatility|Brent root-finding with a round-trip recovery test|
|VC application|ESOP/warrant valuation and its sensitivity to the volatility assumption|

## Key results (S=K=100, T=1, r=5%, σ=20%)

* Call = **10.4506**, Put = **5.5735**; put–call parity holds to machine precision
* Monte Carlo (200k paths) agrees with the closed form within its 95% confidence interval; antithetic sampling lowers the standard error
* Analytical Greeks match central finite differences to 6 decimal places
* Implied volatility is recovered to \~1e-14

## Run it

**Google Colab (no setup):** click the badge above, or paste `src/black\_scholes.py` into a single Colab cell and run.

**Locally:**

```bash
git clone https://github.com/Abhimanyu-108/black-scholes-option-pricing.git
cd black-scholes-option-pricing
pip install -r requirements.txt
jupyter notebook notebooks/Black\_Scholes\_Option\_Pricing.ipynb
```

## Model assumptions \& limitations

GBM underlying, constant volatility and rate, no arbitrage, European exercise. Real markets show a volatility smile, jumps and early exercise. For private-company ESOPs, analysts also apply a discount for lack of marketability (DLOM) and adjust for vesting. The VC example uses **illustrative inputs**, not data from any real company.

## Roadmap

* \[ ] Binomial tree for American options
* \[ ] Heston stochastic-volatility model
* \[ ] Implied-volatility surface from real option-chain data
* \[ ] DLOM-adjusted ESOP valuation

## Author

**Abhimanyu Gaur** — B.Sc. Statistics (Hons.), Ramjas College, University of Delhi · interested in venture capital and fintech.
Educational project; not investment advice.

## License

MIT

