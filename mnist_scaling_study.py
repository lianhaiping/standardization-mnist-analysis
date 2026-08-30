import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Always standardize your ML features?

        A reproducible study on MNIST. Pixels are divided by 255 so every feature
        already lives in [0, 1] — the open question is whether `StandardScaler`
        (centering + unit variance) earns its place on top of that.

    
        *Tested on Python 3.12.3, scikit-learn 1.8.0, numpy 2.4.4, marimo 0.24.0.
    
        """
    )
    return (mo,)


@app.cell
def _():
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_predict,
        cross_val_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    RANDOM_STATE = 0
    return (
        LogisticRegression,
        MLPClassifier,
        RANDOM_STATE,
        SGDClassifier,
        StandardScaler,
        StratifiedKFold,
        VarianceThreshold,
        cross_val_predict,
        cross_val_score,
        make_pipeline,
        np,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Load MNIST

    Fetched from OpenML as CSV

    I shuffle the 60k training set with a fixed seed and take 50k from it,
    so the sample does not depend on the file's row order.

    Everything reported here is cross-validated within that 50k.
    """)
    return


@app.cell
def _(np):
    def load_mnist():
        """Return (X_train, y_train, X_test, y_test), pixels scaled to [0, 1].

        """
        from sklearn.datasets import fetch_openml

        Xf, yf = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        Xf = (Xf / 255.0).astype(np.float32)
        yf = yf.astype(np.int64)
        return Xf[:60000], yf[:60000], Xf[60000:], yf[60000:]

    X_gray, y_train_full, X_gray_test, y_test = load_mnist()
    return X_gray, y_train_full


@app.cell
def _(RANDOM_STATE, X_gray, np, y_train_full):
    N_POOL = 50_000

    # Shuffle the 60k training set with a fixed seed before slicing, so the
    # sample is independent of the file's row order.
    _rng = np.random.RandomState(RANDOM_STATE)
    POOL_IDX = _rng.permutation(len(X_gray))[:N_POOL]

    # Standard grayscale MNIST: pixels already divided by 255, so [0, 1].
    X_train = X_gray[POOL_IDX]
    y_train = y_train_full[POOL_IDX]
    return X_train, y_train


@app.cell
def _(mo):
    mo.md("""
    Check the per-pixel standard deviations.
    """)
    return


@app.cell
def _(StandardScaler, X_train, mo):
    _std = X_train.std(axis=0)
    _scaled = StandardScaler().fit_transform(X_train)

    diagnostics = mo.md(
        f"""
        | statistic | value |
        |---|---|
        | pixels | {X_train.shape[1]} |
        | images | {X_train.shape[0]:,} |
        | max per-pixel std | {_std.max():.4f} |
        | min per-pixel std | {_std.min():.4f} |
        | pixels with std = 0 | {(_std == 0).sum()} |
        | raw value range | [{X_train.min():.1f}, {X_train.max():.1f}] |
        | **standardized range** | **[{_scaled.min():.1f}, {_scaled.max():.1f}]** |

    
        """
    )
    diagnostics
    return


@app.cell
def _(mo):
    mo.md("""
    ## The experiment
    """)
    return


@app.cell
def _(mo):
    n_samples = mo.ui.slider(
        5_000,
        50_000,
        value=50_000,
        step=5_000,
        label="Training images (50k reproduces the published numbers)",
        show_value=True,
    )
    n_folds = mo.ui.slider(2, 5, value=3, label="CV folds", show_value=True)
    n_seeds = mo.ui.slider(
        1, 5, value=3, label="Random seeds (averaged)", show_value=True
    )
    run = mo.ui.run_button(label="Run cross-validation")
    mo.vstack([n_samples, n_folds, n_seeds, run])
    return n_folds, n_samples, n_seeds, run


@app.cell
def _(
    LogisticRegression,
    MLPClassifier,
    SGDClassifier,
    StandardScaler,
    VarianceThreshold,
    make_pipeline,
):
    MODELS = {
        "Logistic reg (lbfgs)": lambda seed: LogisticRegression(max_iter=2000),
        "MLP (Adam)": lambda seed: MLPClassifier(
            (128,), solver="adam", max_iter=40, random_state=seed
        ),
        "SGDClassifier": lambda seed: SGDClassifier(
            max_iter=20, tol=1e-3, random_state=seed
        ),
        "MLP (plain SGD)": lambda seed: MLPClassifier(
            (128,), solver="sgd", max_iter=40, random_state=seed
        ),
    }

    PREPROCESSORS = {
        "raw": None,
        "standardized": lambda: StandardScaler(),
        "center-only": lambda: StandardScaler(with_std=False),
        "varthresh + std": lambda: make_pipeline(
            VarianceThreshold(0.01), StandardScaler()
        ),
    }
    return MODELS, PREPROCESSORS


@app.cell
def _(MODELS, PREPROCESSORS, make_pipeline):
    def build(model_name, prep_name, seed=0):
        """Preprocessing goes INSIDE the pipeline so it is refit per fold.
        """
        estimator = MODELS[model_name](seed)
        prep = PREPROCESSORS[prep_name]
        return estimator if prep is None else make_pipeline(prep(), estimator)

    return (build,)


@app.cell
def _(
    MODELS,
    StratifiedKFold,
    X_train,
    build,
    cross_val_score,
    mo,
    n_folds,
    n_samples,
    n_seeds,
    np,
    run,
    y_train,
):
    mo.stop(not run.value, mo.md("*Press the button to run.*"))

    _n = n_samples.value
    _X, _y = X_train[:_n], y_train[:_n]
    _seeds = list(range(n_seeds.value))

    _rows = []
    _total = len(MODELS) * 2 * len(_seeds)
    with mo.status.progress_bar(total=_total) as _bar:
        for _model in MODELS:
            _row = {"model": _model}
            for _prep in ("raw", "standardized"):
                _per_seed = []
                for _seed in _seeds:
                    _cv = StratifiedKFold(
                        n_folds.value, shuffle=True, random_state=_seed
                    )
                    _s = cross_val_score(
                        build(_model, _prep, _seed), _X, _y, cv=_cv
                    )
                    _per_seed.append(_s.mean())
                    _bar.update()
                _row[_prep] = float(np.mean(_per_seed))
                _row[f"{_prep}_sd"] = float(np.std(_per_seed))
                _row[f"{_prep}_runs"] = _per_seed
            _row["delta"] = 100 * (_row["standardized"] - _row["raw"])
            _row["delta_sd"] = 100 * float(
                np.std(
                    [
                        b - a
                        for a, b in zip(
                            _row["raw_runs"], _row["standardized_runs"]
                        )
                    ]
                )
            )
            _rows.append(_row)

    results = _rows

    mo.md(
        f"**{_n:,} images, {n_folds.value}-fold CV, "
        f"{len(_seeds)} seed(s). Mean ± sd across seeds.**\n\n"
        "| model | raw | standardized | Δ points |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {r['model']} "
            f"| {r['raw']:.4f} ± {r['raw_sd']:.4f} "
            f"| {r['standardized']:.4f} ± {r['standardized_sd']:.4f} "
            f"| {'+' if r['delta'] > 0 else ''}{r['delta']:.2f} "
            f"± {r['delta_sd']:.2f} |"
            for r in results
        )
    )
    return (results,)


@app.cell
def _(mo, results):
    mo.stop(not results)

    _helped = [r["model"] for r in results if r["delta"] > 0]
    _hurt = [r["model"] for r in results if r["delta"] <= 0]

    mo.md(
        f"""
        **Helped by standardizing:** {", ".join(_helped) or "none"}

        **Hurt by standardizing:** {", ".join(_hurt) or "none"}

   
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## But does that mean Adam doesn't need scaling? No.

    Adam conditions *gradients*, not the *forward pass* — it cannot rescue an activation that
    is already saturated or drowned out by a feature on a different scale.

    Below, 100 columns are multiplied by 1000. No information is added or
    removed; only the units change. Uses the same sample size, fold count
    and seeds as the main experiment, so the numbers are directly comparable.

    Run for both an adaptive optimizer (Adam) and a quasi-Newton one
    (L-BFGS).
    """)
    return


@app.cell
def _(mo):
    run_hetero = mo.ui.run_button(
        label="Run heterogeneous-scale demo (uses the sliders above)"
    )
    run_hetero
    return (run_hetero,)


@app.cell
def _(
    LogisticRegression,
    MLPClassifier,
    StandardScaler,
    StratifiedKFold,
    X_train,
    cross_val_score,
    make_pipeline,
    mo,
    n_folds,
    n_samples,
    n_seeds,
    np,
    run_hetero,
    y_train,
):
    mo.stop(not run_hetero.value, mo.md("*Press the button to run.*"))

    _n = n_samples.value
    _X, _y = X_train[:_n], y_train[:_n]

    # sabotage 100 columns: multiply by 1000. Same information, different units.
    _rng = np.random.RandomState(0)
    _mult = np.ones(_X.shape[1], dtype=np.float32)
    _mult[_rng.choice(_X.shape[1], 100, replace=False)] = 1000.0
    _H = _X * _mult

    _MK = {
        "MLP-128 (adam)": lambda seed: MLPClassifier(
            (128,), solver="adam", max_iter=40, random_state=seed
        ),
        "LogReg (lbfgs)": lambda seed: LogisticRegression(max_iter=2000),
    }

    _seeds = list(range(n_seeds.value))
    _out = {}
    with mo.status.progress_bar(total=4 * len(_MK) * len(_seeds)) as _bar:
        for _mname, _mk in _MK.items():
            for _fname, _F in (("homogeneous", _X), ("x1000", _H)):
                for _prep in ("raw", "standardized"):
                    _acc = []
                    for _seed in _seeds:
                        _est = _mk(_seed)
                        if _prep == "standardized":
                            _est = make_pipeline(StandardScaler(), _est)
                        _cv = StratifiedKFold(
                            n_folds.value, shuffle=True, random_state=_seed
                        )
                        _acc.append(cross_val_score(_est, _F, _y, cv=_cv).mean())
                        _bar.update()
                    _out[(_mname, _fname, _prep)] = (
                        float(np.mean(_acc)),
                        float(np.std(_acc)),
                    )

    hetero = _out

    def _row(mname, fname):
        r, rsd = hetero[(mname, fname, "raw")]
        s_, ssd = hetero[(mname, fname, "standardized")]
        label = "[0,1] as-is" if fname == "homogeneous" else "100 cols x1000"
        return (
            f"| {mname} | {label} | {r:.4f} ± {rsd:.4f} | "
            f"{s_:.4f} ± {ssd:.4f} | {100 * (s_ - r):+.2f} |"
        )

    mo.md(
        f"**{_n:,} images, {n_folds.value}-fold CV, {len(_seeds)} seed(s) — "
        "same protocol as the table above.**\n\n"
        "| model | features | raw | standardized | Δ points |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            _row(m, f)
            for m in _MK
            for f in ("homogeneous", "x1000")
        )
   
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaway

    In general, standardization helps.

    Always standardize? Too simple.

    Never standardize? Too naive.

    Standardize for what data, what model, what optimizer, and what objective?

    That is the question.
    """)
    return


if __name__ == "__main__":
    app.run()
