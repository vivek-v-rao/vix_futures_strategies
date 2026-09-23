# Volatility and correlation models for VIX futures

The models fitted by `xcontract_stats.py`: what each one says, how it is estimated, and which option
prints it. This is about the **formulas**, not the fitted numbers. Every parameter in here depends on
the sample it was fitted on, and the sample matters more than it usually does — the correlation
across the curve, for one, is materially flatter since 2018 than before. Use `--min-date` and
`--max-date` to fit a window, and read the numbers off your own run rather than off any document.

Two notations run through everything:

| Symbol | Meaning |
|---|---|
| $F$ | a contract's price, in VIX points |
| $\tau$ | trading days from that close to the contract's last close |
| $u$ | a day's change in points, $F_t - F_{t-1}$ |
| $y$ | $\log\lvert u\rvert$, the quantity every scale model is fitted on |
| $d$ | trading days between two contracts' expirations |
| $m$ | trading days the nearer of two contracts has left |
| $\lambda$ | a decay, so that a half-life is $\log 2 / -\log\lambda$ |

Fitting is on $\log\lvert u\rvert$ rather than on $u^2$ throughout. A squared return is a one-day
estimate of a variance and a terrible one; its logarithm is better behaved in a regression, and a
sum of squares is dominated by the few days that dominate the sample.

## Contents

- [Definitions](#definitions)
- [Scale: how big a move is](#scale-how-big-a-move-is)
- [Time variation and asymmetry](#time-variation-and-asymmetry)
- [Shape: what distribution the moves come from](#shape-what-distribution-the-moves-come-from)
- [Other predictors of scale](#other-predictors-of-scale)
- [Carry along the curve](#carry-along-the-curve)
- [Correlation between two contracts](#correlation-between-two-contracts)
- [Estimation notes](#estimation-notes)

## Definitions

These are not models — they are the quantities the models are fitted to, and the ones
`xcurve_stats.py` reports by tenor without fitting anything to them. They are here because the
models below use them and because all three programs mean the same thing by them.

**Tenor.** Tenor 1 is the contract nearest to expiration that can still be held, so a contract at
its last close is not counted and the one behind it moves up that day. Tenor 0, where it appears, is
spot VIX: the thing the curve is a curve of, which cannot be held and so has no return and no carry.

**Returns follow the contract, not the slot.** A day's return at a tenor is the return of the
contract that stood at that tenor on *both* closes, so a roll is a missing day rather than the gap
between two different contracts. A return is computed only when the previous close was on the
previous trading day, so a gap in a contract's history does not become a multi-day return.

**Days to expiry.** Trading days from a close to the contract's last close. A contract that has
settled is counted to its final close, which is known; one still trading is counted to its expiry,
fixed by exchange rule and published years ahead, through a calendar that runs past the data.

**Carry.** For contract $k$ with close $F_k$ and $T_k$ trading days left, against the next contract
$j$ closer to expiration — spot VIX for the front contract, with $T_j = 0$:

$$
\text{carry} = -\frac{F_k - F_j}{(T_k - T_j)\,F_k}
\qquad\text{or}\qquad
-\frac{\log(F_k / F_j)}{T_k - T_j}\ \text{in logs}
$$

the daily return from rolling down the curve if its shape did not change. Negative in contango. It
is the part of a return that can be seen in advance: the price of the insurance for a long, the
premium collected for a short.

**Beta to VIX.** The points a contract moves per point spot VIX moves,
$\mathrm{cov}(r, \Delta\text{VIX}) / \mathrm{var}(\Delta\text{VIX})$ — the number to size
a hedge with. It falls along the curve, as carry does.

**Premium and contango share.** A tenor's average distance above spot in points and as a share of
VIX, and the share of days it closed above the tenor before it.

`xcurve_stats.py` reports all of these by tenor rank, plus correlations between tenors, and fits no
parameters. It is the descriptive counterpart to the models here: same quantities, cut by rank
rather than by days to expiry, which is a different cut because a contract forty days out is the
first or the second depending on where the roll has got to.

## Scale: how big a move is

### Decay with maturity (`--vol-decay`)

Volatility rises as expiration approaches, the Samuelson effect. Three shapes are fitted to the
dispersion of $u$ at each number of days left:

$$
\text{power law: } \sigma = a\,\tau^{-b}
\qquad
\text{exponential: } \sigma = a\,e^{-k\tau}
\qquad
\text{shifted power: } \sigma = a\,(\tau + c)^{-b}
$$

The shift $c$ matters at the short end: a plain power law must diverge at $\tau \to 0$, where the
dispersion in fact turns over. Fitted on a median absolute deviation, with the fit against the
standard deviation reported alongside to show the ranking does not turn on which is used.

Implemented in `dispersion_by_maturity` and `decay_fits`.

### Level and maturity together (`--cev`)

The size of a move depends on the level it moved from as well as on the days left. Three shapes,
each fitted jointly in $\log\lvert u\rvert$:

$$
\text{power: } \lvert u\rvert = a\,F^{\gamma}(\tau + c)^{-b}
$$
$$
\text{displaced: } \lvert u\rvert = a\,(F - \delta)\,(\tau + c)^{-b}
$$
$$
\text{displaced power: } \lvert u\rvert = a\,(F - \delta)^{\gamma}(\tau + c)^{-b}
$$

The first is constant elasticity of variance in the level. The second is a displaced diffusion:
lognormal not in the price but in the excess of the price over a floor $\delta$, which is what a
volatility index has instead of zero. $\gamma = 1$ there is imposed, not fitted.

A displaced model has no single elasticity. Its local elasticity is

$$
\frac{\partial \log \lvert u\rvert}{\partial \log F} = \gamma\,\frac{F}{F - \delta}
$$

which is large when $F$ is near the floor and approaches $\gamma$ when $F$ is far above it. A
constant-elasticity fit to displaced data recovers roughly that elasticity at the middle of the
sample and is wrong at both ends, which is why the two are reported side by side.

**Pooled against within-day.** Each shape is fitted twice. *Pooled* asks whether every contract is
livelier when VIX is high, which is the question a forecast asks. *Within day* takes each trade
date's mean out first and asks whether the dearer contract is the livelier one on the same day,
which is a question about the curve's shape. They need not agree, and reading a pooled elasticity as
if it were a statement about the curve is the trap the two rows exist to prevent.

**Rising floor** (`--rising-floor`). The floor may grow with maturity, since even in a calm market a
long-dated contract is not priced at the near one's level:

$$
\delta(\tau) = \delta_0 + s\sqrt{\tau}
$$

Note this is the floor a *volatility* model wants — the level below which a contract stops moving —
which need not equal the level below which the curve stops reaching.

Implemented in `cev_frame`, `elasticity_fits`, `implied_elasticity`.

## Time variation and asymmetry

### Clustering (`--riskmetrics`)

Whether the size of a move still clusters in time once level and maturity are accounted for. The
RiskMetrics recursion on the residual scale $\varepsilon$, averaged across the contracts trading
each day (clustering is something the market does, not something a contract does):

$$
v_t = \lambda\,v_{t-1} + (1 - \lambda)\,\varepsilon_{t-1}^2
$$

$\log v_t$ then enters the scale regression as another regressor. It uses only days strictly before
the one it speaks for, so it is a forecast rather than a description. `--rm-lambda` sweeps the
memory; `--rm-split` sets the date that divides fitting from scoring.

Implemented in `riskmetrics_factor`, `clustering`, `riskmetrics_forecasts`.

### Asymmetry (`--asymmetry`)

Whether an up move leaves tomorrow livelier than a down move of the same size — the reverse of the
equity leverage effect, VIX futures being lively when VIX rises. Both the size and the sign of
yesterday's market move enter the scale regression:

$$
y_t = \beta_0 + \beta_1 \log(F_{t-1} - \delta) + \beta_2 \log(\tau_t + c)
    + \beta_3 \log v_t + \alpha\,\lvert M_{t-1}\rvert + \mu\,M_{t-1} + \varepsilon_t
$$

with $M$ the market's move. Then $\alpha + \mu$ is what an up move of a given size does to the
expected log size of the next move and $\alpha - \mu$ what a down move does, so $\mu \neq 0$ is the
asymmetry. Both are measured against a day of no move at all, which hardly happens, so their
difference is the quantity worth quoting.

Implemented in `market_moves`, `asymmetry_fits`, `asymmetry_by_move`.

## Shape: what distribution the moves come from

`--innovations` divides each signed move by the scale the model expects of it and asks what the
standardized residual is distributed as. Three candidates, compared by AIC and BIC on the same
residuals, counting only each distribution's own parameters since the scale model is common:

- **Normal**, in the table to be beaten.
- **Student $t$** with $\nu$ fitted, standardized to unit variance.
- **Hansen's skewed $t$**, with a skewness parameter $\eta \in (-1, 1)$ as well as $\nu$:

$$
a = 4\eta c\,\frac{\nu - 2}{\nu - 1},
\qquad
b = \sqrt{1 + 3\eta^2 - a^2},
\qquad
c = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi(\nu-2)}\;\Gamma\!\left(\frac{\nu}{2}\right)}
$$

$$
\log f(z) = \log(bc) - \frac{\nu+1}{2}
  \log\!\left(1 + \frac{1}{\nu - 2}\left(\frac{bz + a}{1 \mp \eta}\right)^{2}\right)
$$

taking $1 - \eta$ when $z < -a/b$ and $1 + \eta$ otherwise. The density is standardized to zero mean
and unit variance, so $\eta$ is a shape and not a location.

`--skew-by-level` fits the tilt inside bands of the price the move started from, to test whether
conditional skewness depends on the level. A parametric level-dependent tilt is deliberately *not*
included: a displacement already does much of that work, since a fall from 20 to 10 and a fall from
40 to 12 are the same fraction of the excess over a floor near 8.75.

Implemented in `skewed_t_logpdf`, `standardized_moves`, `innovation_fits`, `skew_by_level`.

## Other predictors of scale

### Curve slope (`--slope`)

Whether a flat or backwardated curve means livelier futures. Slope over $n$ steps, in logs per step:

$$
S_n = \frac{1}{n}\log\frac{F_n}{\text{VIX}}
$$

positive in contango. Averaging the individual step ratios gives the same thing, the terms
telescoping, so only where the far end is taken matters — which is what `--slope-steps` varies. The
slope is lagged a day and added to the scale regression beside the clustering and asymmetry terms,
since a flat curve and a turbulent week are partly the same weather.

### VVIX (`--vvix`)

Whether the implied volatility of VIX says anything the price of the future does not. $\log$ VVIX,
lagged a day, enters the same regression. Scored before and after a split date, and with the split
reversed, because a regressor with a large mean and a slope that wanders between regimes can help in
sample and hurt out of it.

Implemented in `curve_slopes`, `slope_buckets`, `slope_fits`, `vvix_columns`, `vvix_fits`,
`vvix_by_period`.

## Carry along the curve

Carry is worked out from two prices and the days between them, so its shape along the curve is known
to a precision the realized drift never reaches: a daily return's mean is a small fraction of its
spread, and an exponent fitted to the drift is flat to within noise. The same three shapes as the
volatility decay, but fitted on the level rather than on a logarithm, carry being negative in
contango, so each carries a constant as well:

$$
\text{carry} = a + B(\tau + c)^{-b}
\qquad
a + B e^{-k\tau}
\qquad
a + \frac{B}{\tau + c}
$$

Elasticities are then computed on the fitted curves rather than read off a parameter, since a shape
with an offset has an exponent that is not its rate of fall:

$$
\frac{d \log \lvert\text{carry}\rvert}{d \log \tau}
$$

Comparing this against the same quantity for volatility says how what a unit of risk is paid changes
along the curve — the term-structure form of the Sharpe ratios by tenor.

Implemented in `carry_by_maturity`, `carry_decay_fits`, `decay_elasticities`.

## Correlation between two contracts

### The baseline (`--correlations`)

The starting point is the plain full-sample correlation matrix of daily returns by tenor, with spot
VIX and SPY alongside. Returns are taken only where the same contract stood at a tenor on both
closes, so a roll is a missing day rather than a jump between two contracts. Spot VIX is in it
because the curve is a curve of it, though its percentage change is not a return: the index cannot
be held. SPY is in it because a VIX futures position is usually held against equities.

Everything below is an attempt to improve on this matrix, so it is the thing to beat.

### Why not by tenor rank

A full-sample correlation matrix indexed by tenor rank has to say that the second and third
contracts correlate one way and the first and second another, and then swap the two on expiry day,
though nothing about either contract changed overnight. The trading days between two **expirations**
do not change at a roll, so a model in $d$ is continuous through one. That is the whole reason for
preferring $d$ to rank.

### Parametric decay (`--corr-decay`)

Every shape is written as

$$
\rho = \rho_\infty + (1 - \rho_\infty)\,x
$$

where $x$ falls from 1 to 0 and $\rho_\infty$ is the correlation two contracts never fall below
however far apart they are. The floor enters linearly and is solved for by weighted least squares
rather than searched, and is held in $[0, 1)$ — left free it wanders to large negative values that a
shape near 1 everywhere multiplies back into range, which fits as well and means nothing.

| Shape | $x$ | |
|---|---|---|
| exponential | $e^{-\beta d}$ | decay in separation alone |
| stretched | $e^{-\beta d^{g}}$ | the same, with the decay allowed to bend |
| maturity ratio | $(1 + d/m)^{-\beta}$ | decay in the ratio of the two maturities |
| Rebonato | $e^{-\beta d\,e^{-\alpha m}}$ | separation decaying more slowly further out |
| humped | $e^{-\beta d\,\exp(\alpha_1 u + \alpha_2 u^2)},\; u = \log(m/21)$ | decay rate quadratic in log maturity |

The maturity-ratio form is decay in log separation, which gets a maturity effect for free: two
contracts 60 and 80 days out are a smaller ratio apart than two contracts 1 and 21 days out, though
both pairs are a month apart. The Rebonato form is the same thought as the LIBOR market model
writes it, the question there being the correlation of two forward rates.

The humped form exists because neither of those can bend twice. A decay rate that only slows with
maturity cannot represent a maturity effect that rises and then falls, and $\alpha_2 > 0$ in the
humped form puts the loosest decay — the highest correlation — at an interior maturity
$m^{*} = 21\exp(-\alpha_1 / 2\alpha_2)$.

Shapes are fitted to correlations computed in (separation, maturity) cells, weighted by the pairs
behind each cell, with $R^2$ measured against the weighted mean — the single number a full-sample
matrix would use — so it reads as what a shape adds to assuming one correlation for every pair.

Implemented in `contract_pairs`, `correlation_cells`, `correlation_shapes`, `correlation_fits`,
`correlation_surface`.

### Decayed correlation (`--corr-ewma`)

RiskMetrics applied to **separations** rather than to contracts, so the roll cannot disturb it. For
each separation, the day's average cross product and second moment across the pairs at that
separation are decayed separately and then divided:

$$
C_t = \lambda C_{t-1} + (1-\lambda)\,\overline{z_i z_j}\big|_{t-1},
\qquad
V_t = \lambda V_{t-1} + (1-\lambda)\,\overline{\tfrac{1}{2}(z_i^2 + z_j^2)}\big|_{t-1},
\qquad
\rho_t = \frac{C_t}{V_t}
$$

Decaying numerator and denominator separately, rather than decaying the day's ratio, is what keeps
it a correlation: a day of large moves should count for more than a day of small ones, and dividing
first would give them equal say.

**Why pooled and not per contract pair.** Estimating a named pair's correlation from its own history
is what one does for two stocks, and it does not work here. Two stocks have companies behind them,
so their covariance has an idiosyncratic part worth estimating from their joint history; two VIX
futures are two points on one curve, and once separation and maturity are known there is nothing
left over. And there is no time: a named pair coexists for a median of about 74 trading days, which
is shorter than a sensible half-life. `--pair-matrix` reports realized correlations for named
contracts as a *description* of what a book did, with the days behind every cell printed beside it,
and should not be read as a forecast.

### Scoring, and mixing the two

Correlations are scored by the mean log likelihood a bivariate normal over that correlation gives
the moves that followed,

$$
\log L = -\tfrac{1}{2}\left[\log(1 - \rho^2)
  + \frac{z_i^2 + z_j^2 - 2\rho z_i z_j}{1 - \rho^2}\right]
$$

rather than by $R^2$ on a cross product, which is nearly all noise. The likelihood punishes a
correlation for being too high and for being too low, and by the right amount in each direction.

Two ways of combining a decayed estimate with a fitted shape:

$$
\text{weighted: } \rho = w\,\rho_{\text{ewma}} + (1-w)\,\rho_{\text{shape}}
\qquad
\text{tilted: } \rho = \rho_{\text{ewma}}\cdot
  \frac{\rho_{\text{shape}}(d, m)}{\rho_{\text{shape}}(d, \bar m_d)}
$$

The weighted form shrinks the decayed estimate towards the fitted surface. The tilted form instead
takes the *level* at each separation from the decayed estimate and only the *maturity tilt* from the
shape, which is the division of labour the two are suited to: a decayed average of a separation
pools all maturities together and so is blind to the maturity effect, while a shape fitted once is
blind to a change of regime.

A useful check: run the tilt against a shape with no maturity term, such as the exponential. Its
tilt ratio is then 1 everywhere and the tilted row must reproduce the plain decayed row exactly. If
it does not, something is wrong with the implementation and not with the market.

Implemented in `pair_moments`, `ewma_correlation`, `pair_loglik`, `correlation_forecasts`.

## Estimation notes

**Clustered standard errors.** Forty thousand contract-days are not forty thousand pieces of news:
every contract listed moves on the same headline, so the independent unit is the trade date. Errors
are clustered there throughout, and treating the rows as independent roughly halves them.

**Grid search.** Several models have a parameter that enters non-linearly beside others that do not
— a floor, an offset, an exponent. These are found by searching a declared grid whole and then
narrowing around the winner for a round or two, rather than by refining from the start, so the
refinement can only sharpen an optimum the grid already had and never lose one. Parameters that come
back sitting on the edge of their grid should be treated as unidentified rather than as estimates.

**Out of sample.** Anything that could be fitted to noise is scored on days it was not fitted on,
and where the direction of the split could matter it is reversed as well. A model that looks good in
sample and bad out of it is reported as such rather than quietly dropped.

**What the data can and cannot say.** Days on which a contract did not move are dropped by the scale
models, having no logarithm. That removes some stale far-dated closes, so thin trading at the long
end is reduced but not eliminated as an explanation for anything found there.

## Reproducing

```bash
python xcontract_stats.py                         # the tables that describe the contracts
python xcontract_stats.py --cev --rising-floor    # one model at a time
python xcontract_stats.py --corr-decay --corr-ewma --min-date 2018-01-01
python xcontract_stats.py --terse                 # tables named, not read
```

Every option takes an optional filename to write its table to. `--min-date` and `--max-date` narrow
the window before anything is computed. `--notime` suppresses the timing table each run ends with.

## Further reading

- Samuelson, P. (1965). "Proof that properly anticipated prices fluctuate randomly." The maturity
  effect in the first section.
- Rebonato, R. (2004). *Volatility and Correlation*. The correlation forms and the argument for
  writing them with a floor.
- Hansen, B. (1994). "Autoregressive conditional density estimation." The skewed $t$.
- J.P. Morgan (1996). *RiskMetrics Technical Document*. The decayed variance recursion.
