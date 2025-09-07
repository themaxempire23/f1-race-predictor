# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import plotly.express as px
import fastf1
import requests
from urllib.parse import urlencode

# -------------------------
# App config & paths
# -------------------------
st.set_page_config(page_title="F1 Race Predictor", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

st.title("🏁 F1 Race Predictor — Weekend Overview & Simulation")

# Placeholders declared early
wk_holder = st.empty()
sim_holder = st.empty()

# -------------------------
# Cached loaders & helpers
# -------------------------
@st.cache_resource
def load_models():
    reg = load(MODELS / "rf_finishpos.joblib")
    cls = load(MODELS / "rf_top10.joblib")
    # feature list the model was trained with (from the imputer)
    try:
        feat = list(reg.named_steps["imp"].feature_names_in_)
    except Exception:
        feat = None
    return reg, cls, feat

@st.cache_data
def load_weekend(season: int, rnd: int) -> pd.DataFrame:
    path = PROCESSED / f"features_weekend_ext_target_{season}_R{rnd}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Build Phase 1 in notebooks or use 'Build data for this weekend' below."
        )
    wk_ = pd.read_csv(path)
    # Derived feature used in training
    if "grid_drop" not in wk_.columns and {"grid_pos", "qual_pos"}.issubset(wk_.columns):
        wk_["grid_drop"] = pd.to_numeric(wk_["grid_pos"], errors="coerce") - pd.to_numeric(wk_["qual_pos"], errors="coerce")
    return wk_

@st.cache_data
def load_season_enriched(season: int) -> pd.DataFrame:
    p = PROCESSED / f"season_{season}_features_enriched.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p.name}. Run Phase 2 for this season in the notebooks.")
    df = pd.read_csv(p)
    if "grid_drop" not in df.columns and {"grid_pos", "qual_pos"}.issubset(df.columns):
        df["grid_drop"] = pd.to_numeric(df["grid_pos"], errors="coerce") - pd.to_numeric(df["qual_pos"], errors="coerce")
    return df

def align_features(df: pd.DataFrame, feat_list: list[str]) -> pd.DataFrame:
    # Pipeline imputer handles NaN; we just need the exact columns in the same order
    return df.reindex(columns=feat_list)

def points_from_pos(p: int) -> int:
    table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    return table.get(int(p), 0)

def blend_dnf(season_df: pd.DataFrame, wk: pd.DataFrame) -> np.ndarray:
    # conservative DNF flag if not present
    if "dnf" not in season_df.columns:
        def is_dnf_row(row):
            stt = str(row.get("Status", "")).lower()
            pos = row.get("finish_pos", np.nan)
            pts = row.get("points", 0.0)
            return (("finished" not in stt) and ("classified" not in stt) and (pd.isna(pos) or pos > 20)) or (pd.isna(pos) and pts == 0)
        season_df = season_df.copy()
        season_df["dnf"] = season_df.apply(is_dnf_row, axis=1).astype(int)

    driver_dnf = season_df.groupby("Driver")["dnf"].mean()
    team_dnf = season_df.groupby("TeamName")["dnf"].mean()

    drivers = wk["Driver"].tolist()
    teams = wk["TeamName"] if "TeamName" in wk.columns else pd.Series(["?"] * len(drivers))
    p = []
    for d, t in zip(drivers, teams):
        val = 0.5 * driver_dnf.get(d, np.nan) + 0.5 * team_dnf.get(t, np.nan)
        if np.isnan(val):
            val = float(driver_dnf.mean()) if not np.isnan(driver_dnf.mean()) else 0.07
        p.append(float(np.clip(val, 0.01, 0.30)))
    return np.array(p)

def simulate(reg, cls, Xw: pd.DataFrame, wk: pd.DataFrame, p_dnf: np.ndarray, sigma: float,
             N: int, grid_bias: float, sc_prob: float, sc_scale: float, seed: int = 42):
    drivers = wk["Driver"].tolist()
    grid = wk["grid_pos"] if "grid_pos" in wk.columns else pd.Series([np.nan] * len(drivers))

    y_hat = reg.predict(Xw)
    p_pts = cls.predict_proba(Xw)[:, 1]
    grid_v = grid.values if isinstance(grid, pd.Series) else np.array(grid)
    base = y_hat if np.all(np.isnan(grid_v)) else (y_hat + grid_bias * (grid_v - y_hat))

    sims_pos = np.zeros((N, len(drivers)), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(N):
        scf = sc_scale if rng.random() < sc_prob else 1.0
        noise = rng.normal(0, sigma * scf, size=len(drivers))
        score = base + noise
        dnf = rng.random(len(drivers)) < p_dnf
        alive = np.where(~dnf)[0]
        dnf_i = np.where(dnf)[0]
        order_alive = alive[np.argsort(score[alive], kind="mergesort")]
        pos = np.empty(len(drivers), dtype=int)
        pos[order_alive] = np.arange(1, len(order_alive) + 1)
        pos[dnf_i[np.argsort(score[dnf_i], kind="mergesort")]] = np.arange(len(order_alive) + 1, len(drivers) + 1)
        sims_pos[_] = pos

    exp_pos = sims_pos.mean(axis=0)
    exp_pts = np.apply_along_axis(lambda col: np.mean([points_from_pos(p) for p in col]), 0, sims_pos)
    out = pd.DataFrame({
        "Driver": drivers,
        "grid_pos": grid.values if isinstance(grid, pd.Series) else grid,
        "pred_base_pos": y_hat.round(2),
        "exp_pos": exp_pos.round(2),
        "p_win": (sims_pos == 1).mean(axis=0).round(3),
        "p_podium": (sims_pos <= 3).mean(axis=0).round(3),
        "p_top10_sim": (sims_pos <= 10).mean(axis=0).round(3),
        "p_top10_cls": p_pts.round(3),
        "exp_points": exp_pts.round(2),
    }).sort_values(["exp_pos", "pred_base_pos"]).reset_index(drop=True)

    return out, sims_pos

# -------------------------
# Minimal builder (FastF1) + OpenF1 fallback for recent rounds
# -------------------------
def _load_results_fast(year: int, rnd: int, code: str):
    sess = fastf1.get_session(year, rnd, code)
    try:
        sess.load(laps=False, telemetry=False, weather=False, messages=False)
    except TypeError:
        try:
            sess.load()
        except Exception:
            sess.load(telemetry=False)
    return sess

def _openf1_get_json(endpoint: str, **params) -> list:
    base = "https://api.openf1.org/v1/" + endpoint
    url = base + "?" + urlencode(params)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _openf1_session_keys_by_round(year: int, rnd: int) -> tuple[int | None, int | None]:
    """Map year+round → (q_key, r_key) using OpenF1 sessions.
    Strategy: list all Qualifying (year) sorted by date → take nth; same for Race.
    """
    sess_q = _openf1_get_json("sessions", year=year, session_name="Qualifying")
    sess_r = _openf1_get_json("sessions", year=year, session_name="Race")

    def pick_nth(sessions: list, n: int) -> int | None:
        if not sessions:
            return None
        df = pd.DataFrame(sessions)
        if "date_start" not in df or "session_key" not in df:
            return None
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        df = df.sort_values("date_start").reset_index(drop=True)
        if 0 <= (n - 1) < len(df):
            return int(df.loc[n - 1, "session_key"])
        return None

    return pick_nth(sess_q, rnd), pick_nth(sess_r, rnd)

def build_minimal_weekend_features(season: int, rnd: int) -> Path:
    """
    Build features_weekend_ext_target_<season>_R<rnd>.csv
    1) Try FastF1 (results-only).
    2) If FastF1 race/quali results missing (common for very recent rounds),
       fall back to OpenF1:
         - sessions?year=Y&session_name=Qualifying/Race → session_key
         - starting_grid?session_key=... (grid)
         - session_result?session_key=... (qual position)
         - laps?session_key=... (optional: best quali lap → delta to pole)
    """
    # ---------- FastF1 attempt
    try:
        q = _load_results_fast(season, rnd, "Q")
        r = _load_results_fast(season, rnd, "R")

        qres = q.results[["Abbreviation", "Position", "Q1", "Q2", "Q3"]].copy()
        if len(qres) == 0:
            raise ValueError("Empty FastF1 quali results")

        qres.columns = ["Driver", "qual_pos", "Q1", "Q2", "Q3"]
        for c in ["Q1", "Q2", "Q3"]:
            qres[c] = pd.to_timedelta(qres[c], errors="coerce")
        qres["best_qual_t"] = qres[["Q1", "Q2", "Q3"]].min(axis=1, skipna=True)
        pole = qres["best_qual_t"].min()
        qres["delta_to_pole_s"] = (qres["best_qual_t"] - pole).dt.total_seconds()
        qres = qres[["Driver", "qual_pos", "delta_to_pole_s"]]

        rres = r.results[["Abbreviation", "TeamName", "GridPosition"]].copy()
        if len(rres) == 0:
            raise ValueError("Empty FastF1 race grid")

        rres.columns = ["Driver", "TeamName", "grid_pos"]

        wk = pd.merge(qres, rres, on="Driver", how="outer")
        wk["qual_pos"] = pd.to_numeric(wk["qual_pos"], errors="coerce")
        wk["grid_pos"] = pd.to_numeric(wk["grid_pos"], errors="coerce")
        wk["grid_drop"] = wk["grid_pos"] - wk["qual_pos"]

        out_path = PROCESSED / f"features_weekend_ext_target_{season}_R{rnd}.csv"
        wk.to_csv(out_path, index=False)
        return out_path

    except Exception:
        # ---------- OpenF1 fallback
        q_key, r_key = _openf1_session_keys_by_round(season, rnd)
        if q_key is None or r_key is None:
            raise RuntimeError("OpenF1 fallback failed to find session keys.")

        # Drivers (to map driver_number -> acronym & team)
        drv_q = _openf1_get_json("drivers", session_key=q_key)
        drv_r = _openf1_get_json("drivers", session_key=r_key)
        df_dq = pd.DataFrame(drv_q)
        df_dr = pd.DataFrame(drv_r)

        # Qualifying position from session_result
        sr = _openf1_get_json("session_result", session_key=q_key)
        df_sr = pd.DataFrame(sr) if sr else pd.DataFrame(columns=["driver_number", "position"])
        df_sr = df_sr.rename(columns={"position": "qual_pos"})

        # Optional: best quali lap per driver → delta to pole
        laps = _openf1_get_json("laps", session_key=q_key)
        if laps:
            df_l = pd.DataFrame(laps)
            if {"driver_number", "lap_duration"}.issubset(df_l.columns):
                best = (df_l.dropna(subset=["lap_duration"])
                          .groupby("driver_number", as_index=False)["lap_duration"].min())
                pole = best["lap_duration"].min() if not best.empty else np.nan
                best["delta_to_pole_s"] = best["lap_duration"] - pole
            else:
                best = pd.DataFrame(columns=["driver_number", "delta_to_pole_s"])
        else:
            best = pd.DataFrame(columns=["driver_number", "delta_to_pole_s"])

        # Starting grid from race session
        sg = _openf1_get_json("starting_grid", session_key=r_key)
        df_sg = pd.DataFrame(sg) if sg else pd.DataFrame(columns=["driver_number", "position"])
        df_sg = df_sg.rename(columns={"position": "grid_pos"})

        # Build driver index with acronym + team
        def drv_map(df):
            # prefer acronym (VER, LEC) and team_name if present
            cols = {}
            if "driver_number" in df.columns: cols["driver_number"] = df["driver_number"]
            if "name_acronym" in df.columns: cols["Driver"] = df["name_acronym"]
            elif "full_name" in df.columns:   cols["Driver"] = df["full_name"]
            if "team_name" in df.columns:     cols["TeamName"] = df["team_name"]
            return pd.DataFrame(cols)

        idx_q = drv_map(df_dq)
        idx_r = drv_map(df_dr)
        idx = pd.concat([idx_q, idx_r], ignore_index=True).drop_duplicates(subset=["driver_number"])

        # Join all
        q_part = pd.merge(df_sr[["driver_number", "qual_pos"]], best[["driver_number", "delta_to_pole_s"]],
                          on="driver_number", how="left")
        r_part = df_sg[["driver_number", "grid_pos"]]

        wk = (idx.merge(q_part, on="driver_number", how="left")
                .merge(r_part, on="driver_number", how="left"))

        # Final clean-up
        wk["qual_pos"] = pd.to_numeric(wk["qual_pos"], errors="coerce")
        wk["grid_pos"] = pd.to_numeric(wk["grid_pos"], errors="coerce")
        wk["grid_drop"] = wk["grid_pos"] - wk["qual_pos"]
        wk = wk.drop(columns=["driver_number"], errors="ignore")

        out_path = PROCESSED / f"features_weekend_ext_target_{season}_R{rnd}.csv"
        wk.to_csv(out_path, index=False)
        return out_path

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Controls")
    season = st.number_input(
        "Season", min_value=2015, max_value=2099,
        value=st.session_state.get("season", 2023), step=1,
        format="%d", key="season"
    )
    round_no = st.number_input(
        "Round", min_value=1, max_value=30,
        value=st.session_state.get("round_no", 1), step=1,
        format="%d", key="round_no"
    )
    sims = st.slider("Simulations (N)", min_value=200, max_value=3000, value=1000, step=100, key="sims")
    grid_bias = st.slider("Grid inertia", 0.0, 0.7, 0.30, 0.05, key="grid_bias")
    sc_prob = st.slider("Safety car prob.", 0.0, 1.0, 0.35, 0.05, key="sc_prob")
    sc_scale = st.slider("SC variance x", 1.0, 2.0, 1.25, 0.05, key="sc_scale")

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        btn_load = st.button("Load weekend", use_container_width=True)
    with cbtn2:
        btn_sim = st.button("Run simulation", use_container_width=True)

    # Build data for selected weekend (fast minimal with OpenF1 fallback)
    if st.button("⚙️ Build data for this weekend", use_container_width=True):
        with st.spinner("Building minimal weekend features..."):
            try:
                outp = build_minimal_weekend_features(int(season), int(round_no))
                st.success(f"Built: {outp.name}")
            except Exception as e:
                st.error(f"Build failed: {e}")

    # Jump to latest completed round (type-safe)
    def _jump_to_latest():
        season_val = int(st.session_state["season"])
        sched = fastf1.get_event_schedule(season_val, include_testing=False)
        done = sched[sched["EventDate"] <= pd.Timestamp.today(tz=None)]
        if done.empty:
            st.info("No completed rounds found for that season.")
            return
        latest = int(done["RoundNumber"].max())
        st.session_state["round_no"] = latest
        st.rerun()

    st.button("Jump to latest completed round", on_click=_jump_to_latest, use_container_width=True)

# -------------------------
# Load models
# -------------------------
try:
    reg_pipe, cls_pipe, features_model = load_models()
except FileNotFoundError as e:
    st.error(f"Model files not found in /models. {e}")
    st.stop()

if features_model is None:
    st.error("Model feature list missing from pipeline. Re-train in 03_modeling_baselines.")
    st.stop()

# -------------------------
# Weekend Overview (Preliminary dashboards)
# -------------------------
if btn_load:
    try:
        wk = load_weekend(int(season), int(round_no))
        season_df = load_season_enriched(int(season))
    except FileNotFoundError as e:
        wk_holder.error(str(e))
    else:
        with wk_holder.container():
            st.subheader(f"Weekend overview — {int(season)} R{int(round_no)}")

            # KPI row
            n_drv = wk["Driver"].nunique() if "Driver" in wk else 0
            avg_fp = wk["fp_mean_all_s"].mean() if "fp_mean_all_s" in wk else np.nan
            med_delta = wk["delta_to_pole_s"].median() if "delta_to_pole_s" in wk else np.nan
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Drivers", n_drv)
            c2.metric("Avg FP mean (s)", f"{avg_fp:.3f}" if pd.notna(avg_fp) else "—")
            c3.metric("Median Δ to pole (s)", f"{med_delta:.3f}" if pd.notna(med_delta) else "—")
            c4.metric("Grid known?", "Yes" if "grid_pos" in wk.columns and wk["grid_pos"].notna().any() else "No")

            # Tables: practice / quali / grid
            cols = st.columns([1.2, 1, 1])
            with cols[0]:
                st.markdown("**Practice pace (lower is faster)**")
                prac_cols = [c for c in ["fp_mean_all_s", "fp_median_longrun_s", "fp_total_laps"] if c in wk.columns]
                if prac_cols:
                    st.dataframe(wk[["Driver"] + prac_cols].sort_values(prac_cols[0]), width="stretch")
                else:
                    st.info("Practice summary not available in minimal build.")
            with cols[1]:
                st.markdown("**Qualifying**")
                q_cols = [c for c in ["qual_pos", "delta_to_pole_s"] if c in wk.columns]
                if q_cols:
                    st.dataframe(wk[["Driver"] + q_cols].sort_values("qual_pos"), width="stretch")
                else:
                    st.info("Qualifying summary not available.")
            with cols[2]:
                st.markdown("**Starting grid**")
                if "grid_pos" in wk.columns:
                    st.dataframe(wk[["Driver", "grid_pos"]].sort_values("grid_pos"), width="stretch")
                else:
                    st.info("Grid not available.")

            # Quick Plotly charts
            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                if {"fp_mean_all_s", "fp_total_laps"}.issubset(wk.columns):
                    fig = px.scatter(
                        wk, x="fp_total_laps", y="fp_mean_all_s", text="Driver",
                        title="Practice: laps vs mean pace (lower is better)"
                    )
                    fig.update_traces(textposition="top center")
                    st.plotly_chart(fig, use_container_width=True)
                elif {"qual_pos", "delta_to_pole_s"}.issubset(wk.columns):
                    fig = px.bar(
                        wk.sort_values("qual_pos"),
                        x="Driver", y="delta_to_pole_s",
                        title="Qualifying: Δ to pole (s)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with g2:
                if "grid_pos" in wk.columns:
                    fig = px.bar(
                        wk.sort_values("grid_pos"),
                        x="Driver", y="grid_pos",
                        title="Starting grid (lower = closer to front)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Track overlays from /reports (if present)
            drs_png = REPORTS / f"drs_overlay_{int(season)}_R{int(round_no)}.png"
            trap_png = REPORTS / f"trap_brake_{int(season)}_R{int(round_no)}.png"
            drs_csv = REPORTS / f"drs_zones_{int(season)}_R{int(round_no)}.csv"

            st.markdown("---")
            st.markdown("### Track overlays")
            t1, t2 = st.columns(2)
            with t1:
                if drs_png.exists():
                    st.image(str(drs_png), caption="DRS zones overlay", use_column_width=True)
                else:
                    st.info("DRS overlay not found — generate via 04_simulation notebook (V1–V6 cells).")
            with t2:
                if trap_png.exists():
                    st.image(str(trap_png), caption="Speed trap & braking hotspots", use_column_width=True)
                else:
                    st.info("Trap/brake overlay not found — generate via 04_simulation notebook.")
            if drs_csv.exists():
                st.markdown("**DRS zones (approx.)**")
                st.dataframe(pd.read_csv(drs_csv), width="stretch")

# -------------------------
# Simulation
# -------------------------
if btn_sim:
    try:
        wk = load_weekend(int(season), int(round_no))
    except FileNotFoundError as e:
        sim_holder.error(str(e))
    else:
        # Align features
        try:
            reg_pipe, cls_pipe, features_model = load_models()
        except Exception as e:
            sim_holder.error(f"Model load failed: {e}")
            st.stop()

        Xw = align_features(wk, features_model)

        # Try current season for sigma + DNF, else previous season, else defaults
        def _try_load_enriched(y):
            try:
                return load_season_enriched(int(y))
            except FileNotFoundError:
                return None

        season_df = _try_load_enriched(int(season))
        fallback_df = _try_load_enriched(int(season) - 1)

        def _compute_sigma(df_for_sigma):
            Xs_ = align_features(df_for_sigma, features_model)
            y_  = df_for_sigma["finish_pos"].astype(float).values
            pred_ = reg_pipe.predict(Xs_)
            mae_ = float((abs(y_ - pred_)).mean())
            return float(mae_ / 0.798)  # MAE ≈ sigma * sqrt(2/pi)

        if season_df is not None:
            sigma = _compute_sigma(season_df)
            p_dnf = blend_dnf(season_df, wk)
        elif fallback_df is not None:
            sigma = _compute_sigma(fallback_df)
            p_dnf = blend_dnf(fallback_df, wk)
            st.warning("No enriched file for this season yet — using last season to calibrate spread (σ) and DNF.")
        else:
            sigma = 1.8
            p_dnf = np.full(len(wk), 0.08)  # conservative default
            st.warning("No season history found — using defaults (σ≈1.8, DNF≈8%).")

        # Run simulation
        out, sims_pos = simulate(
            reg_pipe, cls_pipe, Xw, wk, p_dnf, sigma,
            N=int(sims), grid_bias=float(grid_bias), sc_prob=float(sc_prob), sc_scale=float(sc_scale)
        )

        with sim_holder.container():
            st.subheader("Model prediction & simulation")
            st.dataframe(out, width="stretch")

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
                file_name=f"sim_{int(season)}_R{int(round_no)}_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
