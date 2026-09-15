import numpy as np
import pandas as pd

from unis.gravity.eventstudy import fit_year_path, joint_wald, year_path


def synthetic_panel(effect: float = -0.7) -> pd.DataFrame:
    """Origins x 3 destinations x 1834-1839, with Göttingen flows cut after 1837."""
    rng = np.random.default_rng(0)
    rows = [(o, d, y) for o in range(60) for d in ["goettingen", "a", "b"]
            for y in range(1834, 1840)]
    df = pd.DataFrame(rows, columns=["origin_id", "dest", "first_year"])
    df["log_dist"] = rng.normal(5, 1, len(df))
    df["same_city"] = rng.random(len(df)) < 0.05
    shock = np.where((df["dest"] == "goettingen") & (df["first_year"] > 1837), np.exp(effect), 1.0)
    df["flow"] = rng.poisson(20 * np.exp(1 - 0.3 * df["log_dist"]) * shock)
    return df


def test_reference_year_is_omitted_and_reported_as_zero():
    fit, years = fit_year_path(synthetic_panel(), "goettingen")
    assert years == list(range(1834, 1840))
    assert "got_y1837" not in fit._coefnames

    ev = year_path(fit)
    assert ev.iloc[0].to_dict() == {"year": 1837, "est": 0.0, "se": 0.0}
    assert ev["year"].iloc[1:].is_monotonic_increasing


def test_recovers_a_post_period_drop():
    fit, years = fit_year_path(synthetic_panel(effect=-0.7), "goettingen")
    ev = year_path(fit)
    post = ev[ev["year"] > 1837]["est"].mean()
    pre = ev[ev["year"] < 1837]["est"].mean()
    assert post - pre < -0.4
    post_coefs = [f"got_y{y}" for y in years if y > 1837]
    assert float(joint_wald(fit, post_coefs)["pvalue"]) < 0.01
