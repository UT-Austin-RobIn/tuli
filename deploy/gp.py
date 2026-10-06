"""Exact Matérn-5/2 GP over (target, start-pose) features for GP-UCB start-pose selection.

GP-UCB: Krause & Ong (2011), Eq. (2). Posterior: Rasmussen & Williams (2006), Eqs. (2.25)-(2.26), Alg. 2.1.
kappa=2 is an empirical exploration setting; the theorem's assumptions do not hold here.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

KAPPA = 2.0
ACTIVE_ROWS = 1024
RECENT_ROWS = 256
MAX_FIT_ROWS = 2000
ACTIVE_SEED = 913500000
FIT_SEED = 913600000


def training_seed(stream, seed=1, index=0):
    """Independent RNG stream per (stream, seed, index); seed 1 keeps the original experiment's streams."""
    if seed == 1:
        return stream + index
    return int(np.random.SeedSequence([seed, stream, index]).generate_state(1)[0])


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


class GaussianProcess:
    """Exact ARD Matérn-5/2 regression with a constant prior mean."""

    def __init__(self, length_scales, signal_std, noise_std, prior_mean=0.0):
        self.length_scales = np.asarray(length_scales, dtype=np.float64)
        self.signal_std, self.noise_std, self.prior_mean = float(signal_std), float(noise_std), float(prior_mean)
        self.X = np.empty((0, len(self.length_scales)))
        self.y = np.empty(0)

    def kernel(self, X, Z):
        radius = np.sqrt(5.0) * cdist(np.asarray(X, dtype=np.float64) / self.length_scales,
                                      np.asarray(Z, dtype=np.float64) / self.length_scales)
        return self.signal_std**2 * (1 + radius + radius**2 / 3) * np.exp(-radius)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float)
        covariance = self.kernel(X, X)
        covariance.flat[::len(y) + 1] += self.noise_std**2
        self.L = cholesky(covariance, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y - self.prior_mean, lower=True))
        self.X, self.y = X.copy(), y.copy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        mean = np.full(len(X), self.prior_mean)
        variance = np.full(len(X), self.signal_std**2)
        for start in range(0, len(X), 128):
            block = slice(start, start + 128)
            if len(self.y):
                cross = self.kernel(self.X, X[block])
                v = solve_triangular(self.L, cross, lower=True, check_finite=False)
                mean[block] += cross.T @ self.alpha
                variance[block] -= np.einsum("ij,ij->j", v, v)
        if not np.isfinite(mean).all() or np.any(variance < -1e-10):
            raise FloatingPointError("invalid GP posterior")
        return mean, np.sqrt(np.maximum(variance, 0))


def marginal_nll(theta, X, y):
    """Mean negative log marginal likelihood and its gradient in log(length scales, signal, noise)."""
    dimension = X.shape[1]
    scales = np.exp(theta[:dimension])
    signal, noise = np.exp(theta[dimension:])
    distance = np.zeros((len(y), len(y)))
    for d, scale in enumerate(scales):
        distance += ((X[:, d, None] - X[None, :, d]) / scale)**2
    radius = np.sqrt(5 * distance)
    exponential = signal**2 * np.exp(-radius)
    kernel = exponential * (1 + radius + radius**2 / 3)
    covariance = kernel.copy()
    covariance.flat[::len(y) + 1] += noise**2
    L = cholesky(covariance, lower=True, check_finite=False)
    alpha = cho_solve((L, True), y, check_finite=False)
    nll = .5 * y @ alpha + np.log(np.diag(L)).sum() + .5 * len(y) * np.log(2 * np.pi)
    weight = cho_solve((L, True), np.eye(len(y)), check_finite=False) - np.outer(alpha, alpha)
    gradient = np.empty(dimension + 2)
    for d, scale in enumerate(scales):
        derivative = (5 / 3) * exponential * (1 + radius) * ((X[:, d, None] - X[None, :, d]) / scale)**2
        gradient[d] = .5 * np.sum(weight * derivative)
    gradient[dimension] = np.sum(weight * kernel)
    gradient[dimension + 1] = noise**2 * np.trace(weight)
    return nll / len(y), gradient / len(y)


class BoundedGP:
    """GP on a fixed-size coreset: the RECENT_ROWS newest observations plus a random reservoir of older ones."""

    def __init__(self, dimension, seed=1):
        self.values = (np.full(dimension, .2), .2, .43, .25)
        self.X = np.empty((0, dimension))
        self.y = np.empty(0)
        self.priorities = np.empty(0)
        self.seed = seed
        self.rng = np.random.default_rng(training_seed(ACTIVE_SEED, seed))
        self._rebuild()

    def _indices(self):
        count = len(self.y)
        if count <= ACTIVE_ROWS:
            return np.arange(count)
        recent = np.arange(count - RECENT_ROWS, count)
        old_slots = ACTIVE_ROWS - RECENT_ROWS
        old = np.argpartition(self.priorities[:count - RECENT_ROWS], old_slots - 1)[:old_slots]
        return np.sort(np.r_[old, recent])

    def _rebuild(self):
        scales, signal, noise, mean = self.values
        self.active_indices = self._indices()
        self.gp = GaussianProcess(scales, signal, noise, prior_mean=mean)
        if len(self.active_indices):
            self.gp.fit(self.X[self.active_indices], self.y[self.active_indices])

    def observe(self, x, y):
        self.X = np.vstack((self.X, np.asarray(x, dtype=float).reshape(1, -1)))
        self.y = np.append(self.y, float(y))
        self.priorities = np.append(self.priorities, self.rng.random())
        self._rebuild()

    def predict(self, X):
        return self.gp.predict(X)

    def scores(self, X):
        mean, std = self.predict(X)
        return mean + KAPPA * std, mean, std

    def configure(self, values):
        self.values = values
        self._rebuild()

    def save(self, path, **metadata):
        scales, signal, noise, mean = self.values
        temporary = Path(path).with_suffix(".tmp.npz")
        np.savez(temporary, X=self.X, y=self.y, priorities=self.priorities,
                 active_indices=self.active_indices, length_scales=scales,
                 settings=np.array([signal, noise, mean, KAPPA]),
                 training_seed=self.seed, rng_state=json.dumps(self.rng.bit_generator.state),
                 schema_version=2, metadata=json.dumps(metadata, allow_nan=False))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 2 or float(data["settings"][3]) != KAPPA:
                raise ValueError(f"unsupported GP checkpoint {path}")
            model = cls(data["X"].shape[1], seed=int(data["training_seed"]))
            model.values = (data["length_scales"], *map(float, data["settings"][:3]))
            model.X, model.y = data["X"].copy(), data["y"].copy()
            model.priorities = data["priorities"].copy()
            model.rng.bit_generator.state = json.loads(str(data["rng_state"]))
            model._rebuild()
            if not np.array_equal(model.active_indices, data["active_indices"]):
                raise ValueError("checkpoint active posterior is inconsistent")
            return model, json.loads(str(data["metadata"]))


class FeatureMap:
    """GP input: normalized target XY, start-EE XY relative to the target, and start-EE height."""

    def __init__(self, ee_xyz, target_low, target_high):
        ee_xyz = np.asarray(ee_xyz)
        self.target_low = np.asarray(target_low, dtype=float)
        self.target_high = np.asarray(target_high, dtype=float)
        self.ee_low, self.ee_span = ee_xyz.min(0), np.ptp(ee_xyz, axis=0)
        self.rel_low = self.ee_low[:2] - self.target_high
        self.rel_span = self.ee_span[:2] + self.target_high - self.target_low

    def __call__(self, target, ee_xyz):
        target, ee_xyz = np.asarray(target), np.asarray(ee_xyz)
        return np.r_[
            (target - self.target_low) / (self.target_high - self.target_low),
            ((ee_xyz[:2] - target) - self.rel_low) / self.rel_span,
            (ee_xyz[2] - self.ee_low[2]) / self.ee_span[2],
        ]

    def metadata(self):
        return {name: getattr(self, name).tolist() for name in (
            "target_low", "target_high", "ee_low", "ee_span", "rel_low", "rel_span")}


def fit_hyperparameters(model, output):
    """Refit kernel hyperparameters on the oldest 80% of observations; accept only if held-out NLL and MSE improve."""
    count = len(model.y)
    split = int(.8 * count)
    pool = np.arange(split)
    if len(pool) > MAX_FIT_ROWS:
        pool = np.sort(np.random.default_rng(training_seed(FIT_SEED, model.seed, count)).choice(
            pool, MAX_FIT_ROWS, replace=False))
    X, y = model.X[pool], model.y[pool]
    current_scales, current_signal, current_noise, _ = model.values
    candidate_mean = float(np.mean(y))
    bounds = [(np.log(.02), np.log(2.))] * X.shape[1] + [(np.log(.01), np.log(.5)), (np.log(.02), np.log(.6))]
    starts = [np.log(np.r_[current_scales, current_signal, current_noise]),
              np.log(np.r_[current_scales, max(np.std(y), .05), current_noise])]
    fits = [minimize(marginal_nll, start, args=(X, y - candidate_mean), jac=True, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 40, "ftol": 1e-9, "gtol": 1e-5}) for start in starts]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not converged:
        report = {"observations": count, "accepted": False, "reason": "no optimizer start converged"}
        write_json(output, report)
        return report
    tuned = np.exp(min(converged, key=lambda fit: fit.fun).x)

    def metrics(values):
        scales, signal, noise, mean = values
        indices = pool[-min(ACTIVE_ROWS, len(pool)):]
        mean_pred, std = GaussianProcess(scales, signal, noise, prior_mean=mean).fit(
            model.X[indices], model.y[indices]).predict(model.X[split:])
        variance = std ** 2 + noise ** 2
        residual = model.y[split:] - mean_pred
        return {"mse": float(np.mean(residual ** 2)),
                "mean_predictive_nll": float(np.mean(.5 * (np.log(2 * np.pi * variance) + residual ** 2 / variance)))}

    current = model.values
    candidate = (tuned[:-2], float(tuned[-2]), float(tuned[-1]), candidate_mean)
    old_metrics, new_metrics = metrics(current), metrics(candidate)
    accepted = (new_metrics["mean_predictive_nll"] < old_metrics["mean_predictive_nll"]
                and new_metrics["mse"] <= old_metrics["mse"])
    if accepted:
        model.configure(candidate)
    report = {"observations": count, "fit_rows": len(pool), "split": split, "accepted": bool(accepted),
              "current": [current[0].tolist(), *current[1:]], "candidate": [candidate[0].tolist(), *candidate[1:]],
              "validation": {"current": old_metrics, "candidate": new_metrics}}
    write_json(output, report)
    return report


def rank_poses(model, feature_map, ee_xyz, target):
    """Start-pose indices from highest to lowest posterior mean (ties: lower index), and the means."""
    mean, _ = model.predict(np.asarray([feature_map(target, ee) for ee in ee_xyz]))
    return np.lexsort((np.arange(len(mean)), -mean)), mean
