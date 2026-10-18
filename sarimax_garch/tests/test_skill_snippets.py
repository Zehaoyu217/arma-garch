"""Every Python code block in the sarimax-garch skill must run as written.

The blocks are executed in document order (SKILL.md, then references/api.md) in one namespace that starts with
monthly synthetic data: y (Series, month-start DatetimeIndex), X (two macro regressors), X_future (24 future
rows), y_new / X_new (a later stretch of data for the filtering recipe)."""
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from sarimax_garch import _core

SKILL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skills", "sarimax-garch")


def python_blocks(path):
    return re.findall(r"```python\n(.*?)```", open(path).read(), flags=re.S)


@pytest.fixture(scope="module")
def namespace():
    T, H = 200, 24
    idx = pd.date_range("2008-01-01", periods=T + H, freq="MS")
    rng = np.random.default_rng(3)
    macro = np.zeros((T + H, 2))
    for t in range(1, T + H):                                    # two persistent macro drivers
        macro[t] = 0.8 * macro[t - 1] + rng.normal(size=2)
    X_all = pd.DataFrame(macro, index=idx, columns=["unemployment", "gdp_growth"])
    y_all, _, _ = _core.simulate(T + H, 0.1, [0.5], [], 0.15, 0.10, 0.75, dist="normal",
                                 X=X_all.values, gamma=[-0.8, 0.5], seed=4)
    y_all = pd.Series(y_all, index=idx, name="revenue")
    return {"y": y_all.iloc[:T], "X": X_all.iloc[:T], "X_future": X_all.iloc[T:],
            "y_new": y_all.iloc[T - 60:], "X_new": X_all.iloc[T - 60:]}


def test_skill_python_blocks_run(namespace):
    if not os.path.exists(os.path.join(SKILL, "SKILL.md")):
        pytest.skip("skill folder not present")
    import matplotlib
    matplotlib.use("Agg")
    blocks = python_blocks(os.path.join(SKILL, "SKILL.md")) + python_blocks(os.path.join(SKILL, "references", "api.md"))
    assert len(blocks) >= 8
    ns = dict(namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, code in enumerate(blocks):
            try:
                exec(compile(code, f"<skill block {i}>", "exec"), ns)
            except Exception as e:                               # report which block broke
                pytest.fail(f"skill code block {i} failed: {type(e).__name__}: {e}\n---\n{code}")
    # the objects the skill tells the reader to look at exist and have the documented shape
    assert ns["fc"].summary_frame(alpha=0.10).shape == (24, 4)
    assert ns["fc"].conditional_volatility.index.equals(namespace["X_future"].index)
    assert set(ns["crps"].columns) == {"levels", "levels + GARCH", "diffs", "diffs + GARCH"}
    assert ns["cover"].shape == (12, 4) and ns["X_season_future"].shape == (24, 4)
    assert ns["fc_season"].summary_frame().notna().all().all()      # macro + seasonal future exog aligned
    assert ns["new"].bse.isna().all()
