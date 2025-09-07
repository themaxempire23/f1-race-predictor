# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path

from joblib import load
import numpy as np
import plotly.express as px

@st.cache_resource
def load_models():
    reg = load(MODELS / "rf_finishpos.joblib")
    cls = load(MODELS / "rf_top10.joblib")
    # feature list the model was trained with
    try:
        feat = list(reg.named_steps["imp"].feature_names_in_)
    except Exception:
        feat = None
    return reg, cls, feat

@st.cache_data
def load_weekend(season:int, rnd:int) -> pd.DataFrame:
    path = PROCESSED / f"features_weekend_ext_target_{season}_R{rnd}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Build Phase 1 for this weekend.")
    wk = pd.read_csv(path)
    # derived feature used in training
    if "grid_drop" not in wk.columns and {"grid_pos","qual_pos"}.issubset(wk.columns):
        wk["grid_drop"] = pd.to_numeric(wk["grid_pos"], errors="coerce") - pd.to_numeric(wk["qual_pos"], errors="coerce")
    return wk

@st.cache_data
def load_season_enriched(season:int) -> pd.DataFrame:
    p = PROCESSED / f"season_{season}_features_enriched.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p.name}. Run Phase 2 for this season.")
    df = pd.read_csv(p)
    if "grid_drop" not in df.columns and {"grid_pos","qual_pos"}.issubset(df.columns):
        df["grid_drop"] = pd.to_numeric(df["grid_pos"], errors="coerce") - pd.to_numeric(df["qual_pos"], errors="coerce")
    return df

def align_features(df: pd.DataFrame, feat_list: list[str]) -> pd.DataFrame:
    # imputer in the pipeline will handle NaN; we just need exact columns + order
    return df.reindex(columns=feat_list)

st.set_page_config(page_title="F1 Race Predictor", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

st.title("🏁 F1 Race Predictor — Weekend Overview & Simulation")

with st.sidebar:
    st.header("Controls")
    season = st.number_input("Season", min_value=2015, max_value=2099, value=2023, step=1)
    round_no = st.number_input("Round", min_value=1, max_value=30, value=1, step=1)
    sims = st.slider("Simulations (N)", min_value=200, max_value=3000, value=1000, step=100)
    grid_bias = st.slider("Grid inertia", 0.0, 0.7, 0.30, 0.05)
    sc_prob = st.slider("Safety car prob.", 0.0, 1.0, 0.35, 0.05)
    sc_scale = st.slider("SC variance x", 1.0, 2.0, 1.25, 0.05)
    btn_load = st.button("Load weekend")
    btn_sim  = st.button("Run simulation")

    import fastf1
if st.sidebar.button("Jump to latest completed round"):
    sched = fastf1.get_event_schedule(season, include_testing=False)
    done = sched[sched["EventDate"] <= pd.Timestamp.today(tz=None)]
    if not done.empty:
        latest = int(done["RoundNumber"].max())
        st.session_state["season"] = season
        st.session_state["round_no"] = latest
        st.experimental_rerun()
    else:
        st.sidebar.info("No completed rounds found for that season.")



reg_pipe, cls_pipe, features_model = load_models()
if features_model is None:
    st.error("Model feature list missing from pipeline. Re-train in 03_modeling_baselines.")
    st.stop()

if btn_load:
    try:
        wk = load_weekend(season, round_no)
        season_df = load_season_enriched(season)
    except FileNotFoundError as e:
        wk_holder.error(str(e))
    else:
        with wk_holder.container():
            st.subheader(f"Weekend overview — {season} R{round_no}")
            # basic info
            cols = st.columns([1.2, 1, 1])
            with cols[0]:
                # Practice summary (means)
                prac_cols = [c for c in ["fp_mean_all_s","fp_median_longrun_s","fp_total_laps"] if c in wk.columns]
                if prac_cols:
                    st.markdown("**Practice pace (lower is faster)**")
                    st.dataframe(wk[["Driver"] + prac_cols].sort_values(prac_cols[0]))
            with cols[1]:
                # Qualifying summary
                q_cols = [c for c in ["qual_pos","delta_to_pole_s"] if c in wk.columns]
                if q_cols:
                    st.markdown("**Qualifying**")
                    st.dataframe(wk[["Driver"] + q_cols].sort_values("qual_pos"))
            with cols[2]:
                # Grid
                if "grid_pos" in wk.columns:
                    st.markdown("**Starting grid**")
                    st.dataframe(wk[["Driver","grid_pos"]].sort_values("grid_pos"))

            # DRS overlays from /reports (if present)
            drs_png = REPORTS / f"drs_overlay_{season}_R{round_no}.png"
            trap_png = REPORTS / f"trap_brake_{season}_R{round_no}.png"
            drs_csv = REPORTS / f"drs_zones_{season}_R{round_no}.csv"

            st.markdown("---")
            st.markdown("### Track overlays")
            c1, c2 = st.columns(2)
            with c1:
                if drs_png.exists():
                    st.image(str(drs_png), caption="DRS zones overlay")
                else:
                    st.info("DRS overlay not found — run the V1–V6 cells in 04_simulation notebook for this weekend.")
            with c2:
                if trap_png.exists():
                    st.image(str(trap_png), caption="Speed trap & braking hotspots")
                else:
                    st.info("Trap/brake overlay not found — generate via 04_simulation notebook.")
            if drs_csv.exists():
                st.markdown("**DRS zones (approx.)**")
                st.dataframe(pd.read_csv(drs_csv))

import numpy as np

def points_from_pos(p:int) -> int:
    table = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    return table.get(int(p), 0)

def blend_dnf(season_df: pd.DataFrame, wk: pd.DataFrame) -> np.ndarray:
    # conservative DNF flag if not present
    if "dnf" not in season_df.columns:
        def is_dnf_row(row):
            st = str(row.get("Status","")).lower()
            pos = row.get("finish_pos", np.nan)
            pts = row.get("points", 0.0)
            return (("finished" not in st) and ("classified" not in st) and (pd.isna(pos) or pos > 20)) or (pd.isna(pos) and pts==0)
        season_df = season_df.copy()
        season_df["dnf"] = season_df.apply(is_dnf_row, axis=1).astype(int)

    driver_dnf = season_df.groupby("Driver")["dnf"].mean()
    team_dnf   = season_df.groupby("TeamName")["dnf"].mean()

    drivers = wk["Driver"].tolist()
    teams   = wk["TeamName"] if "TeamName" in wk.columns else pd.Series(["?"]*len(drivers))
    p = []
    for d, t in zip(drivers, teams):
        val = 0.5*driver_dnf.get(d, np.nan) + 0.5*team_dnf.get(t, np.nan)
        if np.isnan(val):
            val = float(driver_dnf.mean()) if not np.isnan(driver_dnf.mean()) else 0.07
        p.append(float(np.clip(val, 0.01, 0.30)))
    return np.array(p)

def simulate(reg, cls, Xw: pd.DataFrame, wk: pd.DataFrame, p_dnf: np.ndarray, sigma: float,
             N:int, grid_bias:float, sc_prob:float, sc_scale:float, seed:int=42):
    drivers = wk["Driver"].tolist()
    grid    = wk["grid_pos"] if "grid_pos" in wk.columns else pd.Series([np.nan]*len(drivers))

    y_hat = reg.predict(Xw)
    p_pts = cls.predict_proba(Xw)[:,1]
    grid_v = grid.values if isinstance(grid, pd.Series) else np.array(grid)
    base = y_hat if np.all(np.isnan(grid_v)) else (y_hat + grid_bias*(grid_v - y_hat))

    sims_pos = np.zeros((N, len(drivers)), dtype=int)
    rng = np.random.default_rng(seed)
    for s in range(N):
        scf   = sc_scale if rng.random() < sc_prob else 1.0
        noise = rng.normal(0, sigma*scf, size=len(drivers))
        score = base + noise
        dnf   = rng.random(len(drivers)) < p_dnf
        alive = np.where(~dnf)[0]; dnf_i = np.where(dnf)[0]
        order_alive = alive[np.argsort(score[alive], kind="mergesort")]
        pos = np.empty(len(drivers), dtype=int)
        pos[order_alive] = np.arange(1, len(order_alive)+1)
        pos[dnf_i[np.argsort(score[dnf_i], kind="mergesort")]] = np.arange(len(order_alive)+1, len(drivers)+1)
        sims_pos[s] = pos

    # table
    import numpy as np
    exp_pos = sims_pos.mean(axis=0)
    exp_pts = np.apply_along_axis(lambda col: np.mean([points_from_pos(p) for p in col]), 0, sims_pos)
    out = pd.DataFrame({
        "Driver": drivers,
        "grid_pos": grid.values if isinstance(grid, pd.Series) else grid,
        "pred_base_pos": y_hat.round(2),
        "exp_pos": exp_pos.round(2),
        "p_win": (sims_pos==1).mean(axis=0).round(3),
        "p_podium": (sims_pos<=3).mean(axis=0).round(3),
        "p_top10_sim": (sims_pos<=10).mean(axis=0).round(3),
        "p_top10_cls": p_pts.round(3),
        "exp_points": exp_pts.round(2),
    }).sort_values(["exp_pos","pred_base_pos"]).reset_index(drop=True)

    return out, sims_pos

if btn_sim:
    try:
        wk = load_weekend(season, round_no)
        season_df = load_season_enriched(season)
    except FileNotFoundError as e:
        sim_holder.error(str(e))
    else:
        # align features
        Xw = align_features(wk, features_model)
        Xs = align_features(season_df, features_model)

        # sigma from in-sample residuals
        y_season = season_df["finish_pos"].astype(float).values
        pred_season = reg_pipe.predict(Xs)
        mae_train = float((abs(y_season - pred_season)).mean())
        sigma = float(mae_train / 0.798)

        # DNF blend
        p_dnf = blend_dnf(season_df, wk)

        out, sims_pos = simulate(reg_pipe, cls_pipe, Xw, wk, p_dnf, sigma, sims, grid_bias, sc_prob, sc_scale)

        with sim_holder.container():
            st.subheader("Model prediction & simulation")
            st.dataframe(out, use_container_width=True)

            # Plotly charts
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.bar(out, x="Driver", y="exp_pos", title="Expected finishing position (↓ better)")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.bar(out, x="Driver", y="exp_points", title="Expected points")
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig3 = px.bar(out, x="Driver", y="p_win", title="Win probability")
                st.plotly_chart(fig3, use_container_width=True)
                fig4 = px.bar(out, x="Driver", y="p_podium", title="Podium probability")
                st.plotly_chart(fig4, use_container_width=True)
                fig5 = px.bar(out, x="Driver", y="p_top10_sim", title="Top-10 probability (simulation)")
                st.plotly_chart(fig5, use_container_width=True)

            # Download
            st.download_button(
                "⬇️ Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"sim_{season}_R{round_no}_summary.csv",
                mime="text/csv",
            )



# placeholders
wk_holder = st.empty()
sim_holder = st.empty()
