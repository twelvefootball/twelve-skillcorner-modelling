import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch

DATA_FOLDER = "Data/2024"
PAGE_TITLE = "Pass Suppression Value"
ACCENT = "#22c55e"
ACCENT_ALT = "#38bdf8"
INK = "#0f172a"
MUTED = "#475569"
SURFACE = "#f8fafc"
CARD = "#ffffff"
GRID = "#cbd5e1"
PITCH_GREEN = "#124d36"

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_app_theme():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 24%),
                    linear-gradient(180deg, #f8fbff 0%, #f4f7fb 42%, #eef3f9 100%);
            }}
            .block-container {{
                padding-top: 1.8rem;
                padding-bottom: 2.5rem;
                max-width: 1380px;
            }}
            h1, h2, h3 {{
                color: {INK};
                letter-spacing: -0.02em;
            }}
            div[data-testid="stTabs"] button {{
                font-weight: 600;
            }}
            div[data-testid="stMetric"] {{
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.7rem 0.9rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
            }}
            div[data-testid="stDataFrame"] {{
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.35rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
            }}
            .hero-card {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
                color: white;
                border-radius: 24px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 20px 65px rgba(15, 23, 42, 0.18);
            }}
            .hero-card p {{
                color: rgba(226, 232, 240, 0.9);
                margin: 0.35rem 0 0;
            }}
            .section-card {{
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 22px;
                padding: 1rem 1.1rem 0.9rem;
                margin: 0.55rem 0 1rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
                backdrop-filter: blur(10px);
            }}
            .section-card h3 {{
                margin-bottom: 0.2rem;
            }}
            .section-card p {{
                color: {MUTED};
                margin-bottom: 0;
            }}
            .small-note {{
                color: {MUTED};
                font-size: 0.92rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title, subtitle):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_name(value):
    return value if pd.notna(value) else "Unknown"


def safe_percentile(value):
    return None if pd.isna(value) else f"{value:.0f}th percentile"


def prepare_table(df, decimals_map=None):
    table = df.copy()
    if decimals_map:
        for col, decimals in decimals_map.items():
            if col in table.columns:
                table[col] = table[col].round(decimals)
    return table


# -----------------------------
# Cached data loaders
# -----------------------------
@st.cache_data(ttl=10 * 60)
def get_players():
    return pd.read_csv(f"{DATA_FOLDER}/streamlit/player_stats.csv")


@st.cache_data(ttl=10 * 60)
def get_pair_minutes():
    pair_minutes = pd.read_csv(f"{DATA_FOLDER}/streamlit/pair_minutes.csv")

    def parse_pair(x):
        if isinstance(x, str):
            t = ast.literal_eval(x)
            if len(t) == 2:
                return tuple(sorted(t))
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            return tuple(sorted(x))
        return None

    pair_minutes["pair"] = pair_minutes["pair"].apply(parse_pair)
    return pair_minutes


@st.cache_data(ttl=10 * 60)
def get_player_stats_summary():
    cols = [
        "match_id",
        "Frame",
        "player_id",
        "PSV",
        "PSVrel",
        "ProbsTot",
        "pair",
        "psv_diff",
    ]
    return pd.read_parquet(f"{DATA_FOLDER}/streamlit/model_results.parquet", columns=cols)


@st.cache_data(ttl=10 * 60)
def get_player_pass_summary_data():
    cols = [
        "match_id",
        "Pass_id",
        "player_id",
        "PSV",
        "PSVrel",
        "ProbsTot",
    ]
    return pd.read_parquet(f"{DATA_FOLDER}/streamlit/model_results.parquet", columns=cols)


@st.cache_data(ttl=10 * 60)
def get_player_stats_detail():
    cols = [
        "match_id",
        "Frame",
        "player_id",
        "Pass_id",
        "PSV",
        "PSVrel",
        "ProbsTot",
        "Receiver_ids",
        "PSVraw",
        "PSVrelraw",
        "ProbsTotraw",
        "xT",
        "Probs_cf",
        "Probs",
    ]
    return pd.read_parquet(f"{DATA_FOLDER}/streamlit/model_results.parquet", columns=cols)


@st.cache_data(ttl=10 * 60)
def get_tracking(match_id):
    with open(f"{DATA_FOLDER}/tracking/{match_id}.json") as f:
        return json.load(f)


@st.cache_data(ttl=10 * 60)
def get_meta(match_id):
    with open(f"{DATA_FOLDER}/meta/{match_id}.json") as f:
        return json.load(f)


@st.cache_data(ttl=10 * 60)
def get_match_resources(match_id):
    tracking = get_tracking(match_id)
    meta = get_meta(match_id)

    player_lookup = {
        p["id"]: {"number": p["number"], "team_id": p["team_id"]}
        for p in meta["players"]
    }

    jersey_colors = {
        meta["home_team"]["id"]: meta["home_team_kit"]["jersey_color"],
        meta["away_team"]["id"]: meta["away_team_kit"]["jersey_color"],
    }

    number_colors = {
        meta["home_team"]["id"]: meta["home_team_kit"]["number_color"],
        meta["away_team"]["id"]: meta["away_team_kit"]["number_color"],
    }

    team_names = [meta["home_team"]["name"], meta["away_team"]["name"]]

    return tracking, player_lookup, jersey_colors, number_colors, team_names


# -----------------------------
# Cached preprocessing
# -----------------------------
@st.cache_data(ttl=10 * 60)
def build_player_summary():
    stats = get_player_stats_summary()
    df_players = get_players()
    pair_minutes = get_pair_minutes()

    pair_result = stats.copy()
    
    pair_result['pair'] = pair_result['pair'].apply(tuple)
    pair_result = pair_result.groupby(["pair", "player_id"], as_index=False)["psv_diff"].mean()
    pair_result = pair_result[
        pair_result["pair"].apply(lambda x: isinstance(x, tuple) and len(x) == 2)
    ]

    pair_result = pair_result.merge(pair_minutes, on="pair", how="left")
    pair_result = pair_result[pair_result["minutes"] >= 300]
    pair_result = pair_result.groupby("player_id", as_index=False)["psv_diff"].mean()

    player_totals = stats.groupby("player_id", as_index=False).agg(
        {"PSV": "mean", "PSVrel": "mean", "ProbsTot": "mean"}
    )

    player_totals = player_totals.merge(pair_result, on="player_id", how="left")

    df_merged = df_players.merge(player_totals, on="player_id", how="right")
    df_merged = df_merged[df_merged["minutes_played"] >= 900].copy()
    df_merged["avg_PSV"] = df_merged["PSV"]
    df_merged["avg_PSVrel"] = df_merged["PSVrel"]
    df_merged["avg_ProbsTot"] = df_merged["ProbsTot"]
    df_merged["avg_psv_diff"] = df_merged["psv_diff"]

    metrics = ["avg_PSV", "avg_PSVrel", "avg_ProbsTot", "avg_psv_diff"]
    for col in metrics:
        df_merged[f"{col}_pct"] = df_merged[col].rank(pct=True) * 100

    return df_merged


@st.cache_data(ttl=10 * 60)
def build_pair_summary():
    result = get_player_stats_summary().copy()
    pair_minutes = get_pair_minutes()
    df_players = get_players()[["player_id", "short_name"]].drop_duplicates()

    result['pair'] = result['pair'].apply(tuple)
    result = result.groupby(["pair", "player_id"], as_index=False)["psv_diff"].mean()
    result = result[result["pair"].apply(lambda x: isinstance(x, tuple) and len(x) == 2)]
    result = result.merge(pair_minutes, on="pair", how="left")
    result = result[result["minutes"] >= 300]

    pair_summary = result.groupby("pair", as_index=False).agg(
        psv_diff=("psv_diff", "first"),
        minutes=("minutes", "first"),
    )

    player_name_map = df_players.set_index("player_id")["short_name"].to_dict()
    pair_summary["player_1"] = pair_summary["pair"].str[0]
    pair_summary["player_2"] = pair_summary["pair"].str[1]
    pair_summary["name_1"] = pair_summary["player_1"].map(player_name_map).map(format_name)
    pair_summary["name_2"] = pair_summary["player_2"].map(player_name_map).map(format_name)
    pair_summary["names"] = pair_summary["name_1"] + " & " + pair_summary["name_2"]

    return pair_summary, result


@st.cache_data(ttl=10 * 60)
def get_player_pass_options(selected_id):
    stats = get_player_pass_summary_data()
    player_stats = stats.loc[stats["player_id"].eq(selected_id)]

    player_by_pass = (
        player_stats.groupby(["match_id", "Pass_id"], as_index=False)
        .agg({"PSV": "mean", "PSVrel": "mean", "ProbsTot": "mean"})
        .sort_values("PSV", ascending=False)
    )

    return player_by_pass


@st.cache_data(ttl=10 * 60)
def get_player_pass_frames(selected_id, match_id, pass_id):
    cols = [
        "match_id", "Frame", "player_id", "Pass_id", "Receiver_ids",
        "PSVraw", "PSVrelraw", "ProbsTotraw", "xT", "Probs_cf", "Probs"
    ]
    return pd.read_parquet(
        f"{DATA_FOLDER}/streamlit/model_results.parquet",
        columns=cols,
        filters=[
            ("player_id", "==", selected_id),
            ("match_id", "==", match_id),
            ("Pass_id", "==", pass_id),
        ],
    ).sort_values("Frame")

@st.cache_data(ttl=10 * 60)
def get_pair_pass_summary_data():
    cols = [
        "match_id",
        "Frame",
        "Pass_id",
        "player_id",
        "PSV",
    ]
    return pd.read_parquet(f"{DATA_FOLDER}/streamlit/model_results.parquet", columns=cols)

@st.cache_data(ttl=10 * 60)
def get_pair_pass_options(selected_pair):
    stats = get_pair_pass_summary_data()

    pair = tuple(sorted(selected_pair))
    pair_set = set(pair)
    p1, p2 = pair

    frame_players = (
        stats.groupby(["match_id", "Frame"])["player_id"].unique().reset_index()
    )
    frame_players["player_set"] = frame_players["player_id"].apply(set)
    frame_players = frame_players[
        frame_players["player_set"].apply(lambda s: pair_set.issubset(s))
    ]

    pair_frames = stats.merge(
        frame_players[["match_id", "Frame"]],
        on=["match_id", "Frame"],
        how="inner",
    )

    pair_only = pair_frames[pair_frames["player_id"].isin(pair)].copy()

    pair_frame_psv = (
        pair_only.pivot_table(
            index=["match_id", "Pass_id", "Frame"],
            columns="player_id",
            values="PSV",
            aggfunc="first",
        )
        .reset_index()
    )

    if p1 not in pair_frame_psv.columns:
        pair_frame_psv[p1] = np.nan
    if p2 not in pair_frame_psv.columns:
        pair_frame_psv[p2] = np.nan

    pair_frame_psv["pair_psv_diff"] = (pair_frame_psv[p1] - pair_frame_psv[p2]).abs()

    pair_passes = pair_frame_psv.groupby(["match_id", "Pass_id"], as_index=False).agg(
        pair_psv_diff=("pair_psv_diff", "mean"),
        max_pair_psv_diff=("pair_psv_diff", "max"),
        start_frame=("Frame", "min"),
        end_frame=("Frame", "max"),
        n_frames=("Frame", "nunique"),
    )

    p1_pass = (
        pair_frame_psv.groupby(["match_id", "Pass_id"])[p1].mean().reset_index(name="p1_psv")
    )
    p2_pass = (
        pair_frame_psv.groupby(["match_id", "Pass_id"])[p2].mean().reset_index(name="p2_psv")
    )

    pair_passes = (
        pair_passes.merge(p1_pass, on=["match_id", "Pass_id"], how="left")
        .merge(p2_pass, on=["match_id", "Pass_id"], how="left")
        .sort_values("pair_psv_diff", ascending=False)
    )

    return pair_passes

@st.cache_data(ttl=10 * 60)
def get_pair_pass_frames(selected_pair, match_id, pass_id):
    cols = [
        "match_id",
        "Frame",
        "player_id",
        "Pass_id",
        "PSV",
    ]
    stats = pd.read_parquet(
        f"{DATA_FOLDER}/streamlit/model_results.parquet",
        columns=cols,
        filters=[
            ("match_id", "==", match_id),
            ("Pass_id", "==", pass_id),
        ],
    )
    pair = tuple(sorted(selected_pair))
    pair_set = set(pair)

    selected_frames = stats.loc[
        stats["match_id"].eq(match_id) & stats["Pass_id"].eq(pass_id)
    ].copy()
    frame_players = (
        selected_frames.groupby(["match_id", "Frame"])["player_id"].unique().reset_index()
    )
    frame_players["player_set"] = frame_players["player_id"].apply(set)
    frame_players = frame_players[
        frame_players["player_set"].apply(lambda s: pair_set.issubset(s))
    ]
    return (
        selected_frames.merge(
            frame_players[["match_id", "Frame"]],
            on=["match_id", "Frame"],
            how="inner",
        )
        .sort_values("Frame")
    )


# -----------------------------
# Plotting
# -----------------------------
def plot_radar_updated(players_df, metrics):
    colors = ["#22c55e", "#0ea5e9", "#f97316", "#8b5cf6", "#ef4444", "#f59e0b"]
    labels = [
        "Pass Suppression Value",
        "Pass Suppression",
        "Relative Pass Suppression Value",
        "Pair Dominance",
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(9.6, 7.6), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f3f7fb")

    for i, (_, row) in enumerate(players_df.iterrows()):
        values = row[metrics].values.astype(float)
        values = np.concatenate([values, [values[0]]])
        color = colors[i % len(colors)]

        ax.plot(angles, values, color=color, linewidth=2.8, label=row["short_name"])
        ax.fill(angles, values, color=color, alpha=0.16)
        ax.scatter(angles[:-1], row[metrics].values.astype(float), color=color, s=42, zorder=3)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle) % 360
        if angle_deg == 90:
            label.set_horizontalalignment("left")
        elif angle_deg == 270:
            label.set_horizontalalignment("right")

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{tick}" for tick in [20, 40, 60, 80, 100]], fontsize=9, color=MUTED)
    ax.grid(color=GRID, alpha=0.7, linewidth=0.9)
    ax.spines["polar"].set_visible(False)
    ax.set_title(
        "Positioning Profile",
        fontsize=18,
        fontweight="bold",
        color=INK,
        pad=24,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=min(3, len(players_df)),
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    return fig, ax


def plot_frame(
    metric_value,
    frame_idx,
    receiver_pids,
    tracking,
    player_lookup,
    jersey_colors,
    number_colors,
    team_names,
    selected_id,
    metric_label,
):
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=105,
        pitch_width=68,
        pitch_color=PITCH_GREEN,
        line_color="#ecfeff",
        linewidth=1.4,
    )

    fig, ax = pitch.draw(figsize=(12.5, 8.2))
    fig.set_dpi(300)
    fig.patch.set_facecolor("#ecfdf5")
    ax.set_facecolor(PITCH_GREEN)
    frame = tracking[frame_idx]
    receiver_idx = {pid: i for i, pid in enumerate(receiver_pids)}

    for p in frame["player_data"]:
        player_id = p["player_id"]
        info = player_lookup.get(player_id)
        if info is None:
            continue

        team_id = info["team_id"]
        kit_number = info["number"]
        x, y = p["x"], p["y"]
        jersey = jersey_colors[team_id]
        number_color = number_colors[team_id]

        if player_id == selected_id:
            pitch.scatter(
                x,
                y,
                s=820,
                color="#fde047",
                ax=ax,
                zorder=3,
                edgecolors="#0f172a",
                linewidths=1.8,
                alpha=0.96,
            )

        pitch.scatter(
            x,
            y,
            s=430,
            color=jersey,
            ax=ax,
            zorder=4,
            edgecolors="#0f172a",
            linewidths=1.2,
        )

        pitch.annotate(
            str(kit_number),
            (x, y),
            color=number_color,
            weight="bold",
            fontsize=10,
            ha="center",
            va="center",
            ax=ax,
            zorder=5,
        )

        if player_id in receiver_idx:
            node_idx = receiver_idx[player_id]
            pass_prob = metric_value[node_idx] * 100
            pitch.annotate(
                f"{pass_prob:.1f}",
                (x, y + 2.4),
                color=INK,
                weight="bold",
                fontsize=9.5,
                ha="center",
                va="center",
                ax=ax,
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.24",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                ),
            )

    ball = frame["ball_data"]
    pitch.scatter(
        ball["x"],
        ball["y"],
        s=120,
        color="white",
        edgecolors="#0f172a",
        ax=ax,
        zorder=6,
        linewidths=1.5,
    )

    ax.set_title(
        f"{team_names[0]} vs {team_names[1]}\n{metric_label} at {frame['timestamp']}",
        fontsize=16,
        color=INK,
        pad=16,
        fontweight="bold",
    )

    return fig, ax


def plot_pair_frame(
    frame_idx,
    tracking,
    player_lookup,
    jersey_colors,
    number_colors,
    selected_pair,
    team_names,
    frame_psv_map,
):
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=105,
        pitch_width=68,
        pitch_color=PITCH_GREEN,
        line_color="#ecfeff",
        linewidth=1.4,
    )

    pair_set = set(selected_pair)
    fig, ax = pitch.draw(figsize=(12.5, 8.2))
    fig.set_dpi(300)
    fig.patch.set_facecolor("#eff6ff")
    ax.set_facecolor(PITCH_GREEN)
    frame = tracking[frame_idx]

    for p in frame["player_data"]:
        player_id = p["player_id"]
        info = player_lookup.get(player_id)
        if info is None:
            continue

        team_id = info["team_id"]
        kit_number = info["number"]
        x, y = p["x"], p["y"]
        jersey = jersey_colors[team_id]

        if player_id in pair_set:
            pitch.scatter(
                x,
                y,
                s=820,
                color="#facc15",
                ax=ax,
                zorder=3,
                edgecolors="#0f172a",
                linewidths=1.8,
                alpha=0.96,
            )

        pitch.scatter(
            x,
            y,
            s=430,
            color=jersey,
            ax=ax,
            zorder=4,
            edgecolors="#0f172a",
            linewidths=1.2,
        )

        pitch.annotate(
            str(kit_number),
            (x, y),
            color=number_colors[team_id],
            weight="bold",
            fontsize=10,
            ha="center",
            va="center",
            ax=ax,
            zorder=5,
        )

        if player_id in pair_set and player_id in frame_psv_map:
            psv_value = frame_psv_map[player_id] * 100
            pitch.annotate(
                f"{psv_value:.1f}",
                (x, y + 2.4),
                color=INK,
                weight="bold",
                fontsize=9.5,
                ha="center",
                va="center",
                ax=ax,
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.24",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                ),
            )

    ball = frame["ball_data"]
    pitch.scatter(
        ball["x"],
        ball["y"],
        s=120,
        color="white",
        edgecolors="#0f172a",
        ax=ax,
        zorder=6,
        linewidths=1.5,
    )

    ax.set_title(
        f"{team_names[0]} vs {team_names[1]}\nPair PSV snapshot at {frame['timestamp']}",
        fontsize=16,
        color=INK,
        pad=16,
        fontweight="bold",
    )

    return fig, ax


def plot_pair_dominance(df, highlight_pair=None, highlight_label=None):
    data = df.copy()
    mean = data["psv_diff"].mean()
    std = data["psv_diff"].std()

    if std == 0 or pd.isna(std):
        data["zscore"] = 0.0
    else:
        data["zscore"] = (-1) * (data["psv_diff"] - mean) / std

    data["pair"] = data["pair"].apply(lambda x: tuple(int(v) for v in x))
    if highlight_pair is not None:
        highlight_pair = tuple(int(v) for v in sorted(highlight_pair))

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=300)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    rng = np.random.default_rng(42)
    y_jitter = rng.uniform(-0.1, 0.1, size=len(data))

    ax.axvspan(data["zscore"].min() - 0.4, 0, color="#dbeafe", alpha=0.7, zorder=0)
    ax.axvspan(0, data["zscore"].max() + 0.4, color="#dcfce7", alpha=0.7, zorder=0)
    ax.scatter(
        data["zscore"],
        y_jitter,
        s=82,
        alpha=0.58,
        color=ACCENT_ALT,
        edgecolors="white",
        linewidth=0.7,
        zorder=2,
    )
    ax.axvline(0, linestyle="--", linewidth=1.8, color=MUTED, alpha=0.85, zorder=1)

    if highlight_pair is not None:
        highlight_data = data[data["pair"] == highlight_pair]
        if not highlight_data.empty:
            hx = highlight_data["zscore"].iloc[0]
            hy = 0.0
            ax.scatter([hx], [hy], s=320, color="#facc15", edgecolor=INK, linewidth=1.7, zorder=4)
            ax.scatter([hx], [hy], s=118, color=ACCENT, edgecolor="white", linewidth=1.2, zorder=5)
            ax.annotate(
                highlight_label if highlight_label is not None else f"{highlight_pair[0]} & {highlight_pair[1]}",
                xy=(hx, hy),
                xytext=(0, 18),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", alpha=0.98),
                zorder=6,
            )

    ax.set_title("Centre Back Pair Dominance", fontsize=18, pad=14, weight="bold", color=INK)
    ax.set_xlabel("Dominance z-score", fontsize=12, labelpad=10, color=INK)
    ax.set_yticks([])
    ax.text(
        0.18,
        -0.24,
        "First player dominates",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color=MUTED,
    )
    ax.text(
        0.82,
        -0.24,
        "Second player dominates",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color=MUTED,
    )
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.grid(axis="x", alpha=0.22, color="#94a3b8")
    ax.set_ylim(-0.16, 0.16)
    plt.tight_layout()

    return fig, ax


apply_app_theme()

st.markdown(
    f"""
    <div class="hero-card">
        <h1 style="margin:0;">{PAGE_TITLE}</h1>
        <p>The Pass Suppression Value (PSV) metric quantifies how effectively a defender suppresses the passing options of the ball carrier. Explore defender profiles, inspect pass-level suppression snapshots, and compare centre back pairings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Player Lookup", "Centre Back Pairings Comparison"])


# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    df_merged = build_player_summary()
    metrics_pct = [
        "avg_PSV_pct",
        "avg_ProbsTot_pct",
        "avg_PSVrel_pct",
        "avg_psv_diff_pct",
    ]

    section_intro(
        "Player lookup",
        "Select one or more defenders to compare positioning profiles. Selecting exactly one player unlocks pass-by-pass frame exploration below.",
    )

    selector_df = (
        df_merged[
            [
                "short_name",
                "minutes_played",
                "avg_PSV_pct",
                "avg_PSVrel_pct",
                "avg_ProbsTot_pct",
                "avg_psv_diff_pct",
            ]
        ]
        .sort_values(["minutes_played", "short_name"], ascending=[False, True])
        .rename(
            columns={
                "short_name": "Player",
                "minutes_played": "Minutes",
                "avg_PSV_pct": "PSV pct",
                "avg_PSVrel_pct": "Relative PSV pct",
                "avg_ProbsTot_pct": "Pass Suppression pct",
                "avg_psv_diff_pct": "Pair dominance pct",
            }
        )
    )

    selected_players_state = st.dataframe(
        prepare_table(
            selector_df,
            {
                "Minutes": 0,
                "PSV pct": 0,
                "Relative PSV pct": 0,
                "Pass Suppression pct": 0,
                "Pair dominance pct": 0,
            },
        ),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        height=360,
    )

    selected_rows = selected_players_state["selection"]["rows"]
    selected_players = selector_df.iloc[selected_rows].merge(
        df_merged,
        left_on=["Player", "Minutes"],
        right_on=["short_name", "minutes_played"],
        how="left",
    )

    if len(selected_players) == 0:
        st.info("Choose at least one defender from the table to show the profile view.")
    else:
        overview_cols = st.columns([1.3, 1.3, 1.1, 1.1])
        overview_cols[0].metric("Selected defenders", f"{len(selected_players)}")
        overview_cols[1].metric("Average minutes", f"{selected_players['minutes_played'].mean():,.0f}")
        overview_cols[2].metric(
            "Top PSV percentile",
            safe_percentile(selected_players["avg_PSV_pct"].max()) or "—",
        )
        overview_cols[3].metric(
            "Top dominance percentile",
            safe_percentile(selected_players["avg_psv_diff_pct"].max()) or "—",
        )

        plot_col, summary_col = st.columns([1.6, 1.0])
        with plot_col:
            fig, ax = plot_radar_updated(selected_players, metrics_pct)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with summary_col:
            section_intro(
                "Submetrics explanation",
                """
                <strong>Pass Suppression Value:</strong> Average value of reducing the likelihood of an opponent being targeted by the ball carrier.<br><br>
                <strong>Relative Pass Suppression Value:</strong> The defender's suppression of a passing option is relative to the likelihood of the pass.<br><br>
                <strong>Pass Suppression:</strong> Average reduction in the overall likelihood of an opponent being targeted by the ball carrier without considering how dangerous the passing option is.<br><br>
                <strong>Pair dominance:</strong> The average difference in PSV compared to the defender's centre back partners.
                """,)

        if len(selected_players) > 1:
            st.warning("Select exactly one player in the first table to explore individual passes and frames.")
        else:
            selected_id = selected_players["player_id"].iloc[0]
            player_name = selected_players["short_name"].iloc[0]
            player_by_pass = get_player_pass_options(selected_id)

            section_intro(
                f"Pass explorer for {player_name}",
                "Pick one possession sequence, choose the metric to overlay, and scrub through the available frames.",
            )

            pass_display = (
                player_by_pass.dropna(axis=1, how="all")
                .rename(
                    columns={
                        "match_id": "Match",
                        "Pass_id": "Pass",
                        "PSV": "PSV",
                        "PSVrel": "Relative PSV",
                        "ProbsTot": "Suppression",
                    }
                )
            )

            selected_pass_state = st.dataframe(
                prepare_table(
                    pass_display,
                    {"PSV": 3, "Relative PSV": 3, "Suppression": 3},
                ),
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                use_container_width=True,
                height=300,
            )

            selected_pass_rows = selected_pass_state["selection"]["rows"]
            if len(selected_pass_rows) == 0:
                st.info("Select one pass from the table to show the pitch view.")
            else:
                selected_pass = player_by_pass.iloc[selected_pass_rows]
                selected_match_id = selected_pass["match_id"].iloc[0]
                selected_pass_id = selected_pass["Pass_id"].iloc[0]
                selected_frames = get_player_pass_frames(
                    selected_id=selected_id,
                    match_id=selected_match_id,
                    pass_id=selected_pass_id,
                )

                control_col, info_col = st.columns([1.8, 1.2])
                with control_col:
                    metric_choice = st.radio(
                        "Displayed metric",
                        [
                            "Pass Suppression Value",
                            "Relative Pass Suppression Value",
                            "Pass Suppression",
                            "Probabilities",
                            "Counterfactual Probabilities",
                            "Expected Threat (xT)",
                        ],
                        horizontal=True,
                    )
                with info_col:
                    st.markdown(
                        '<p class="small-note">The labels above potential receivers show the currently selected metric (multiplied by 100) for that frame.</p>',
                        unsafe_allow_html=True,
                    )

                frames = sorted(selected_frames["Frame"].tolist())
                if len(frames) == 1:
                    frame_nr = frames[0]
                    st.caption(f"Only one frame is available for this pass: {frame_nr}")
                else:
                    frame_nr = st.select_slider("Frame", options=frames, value=frames[0])

                selected_frame_row = selected_frames.loc[selected_frames["Frame"].eq(frame_nr)].iloc[0]
                receiver_pids = selected_frame_row["Receiver_ids"]
                match_id = selected_frame_row["match_id"]

                metric_column_map = {
                    "Pass Suppression Value": "PSVraw",
                    "Relative Pass Suppression Value": "PSVrelraw",
                    "Pass Suppression": "ProbsTotraw",
                    "Expected Threat (xT)": "xT",
                    "Counterfactual Probabilities": "Probs_cf",
                    "Probabilities": "Probs",
                }
                metric_value = selected_frame_row[metric_column_map[metric_choice]]

                top_cols = st.columns(4)
                top_cols[0].metric("Match", f"{match_id}")
                top_cols[1].metric("Pass ID", f"{int(selected_pass['Pass_id'].iloc[0])}")
                top_cols[2].metric("Frames in pass", f"{len(frames)}")
                top_cols[3].metric("Current frame", f"{frame_nr}")

                tracking, player_lookup, jersey_colors, number_colors, team_names = get_match_resources(match_id)
                fig, ax = plot_frame(
                    metric_value=metric_value,
                    frame_idx=selected_frame_row["Frame"],
                    receiver_pids=receiver_pids,
                    tracking=tracking,
                    player_lookup=player_lookup,
                    jersey_colors=jersey_colors,
                    number_colors=number_colors,
                    team_names=team_names,
                    selected_id=selected_id,
                    metric_label=metric_choice,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


with tab2:
    pair_summary, pair_result = build_pair_summary()

    section_intro(
        "Centre back pair comparison",
        "Choose one pairing to inspect its dominance positioning across the sample and then drill down to passes where both defenders are present.",
    )

    sorted_pairs = (
        pair_summary.sort_values("psv_diff", ascending=False)[["names", "minutes", "psv_diff", "pair"]]
        .rename(columns={"names": "Pair", "minutes": "Minutes", "psv_diff": "PSV difference"})
    )

    selected_pair_state = st.dataframe(
        prepare_table(
            sorted_pairs.drop(columns=["pair"]),
            {"Minutes": 0, "PSV difference": 6},
        ),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=360,
    )

    selected_pair_rows = selected_pair_state["selection"]["rows"]
    if len(selected_pair_rows) == 0:
        st.info("Select one pairing from the table to open the comparison view.")
    else:
        selected_pair_df = sorted_pairs.iloc[selected_pair_rows]
        selected_pair = tuple(selected_pair_df["pair"].iloc[0])
        highlight_label = selected_pair_df["Pair"].iloc[0]

        metric_cols = st.columns(3)
        metric_cols[0].metric("Selected pair", highlight_label)
        metric_cols[1].metric("Shared minutes", f"{selected_pair_df['Minutes'].iloc[0]:,.0f}")
        metric_cols[2].metric("PSV difference", f"{selected_pair_df['PSV difference'].iloc[0]:.6f}")

        fig, ax = plot_pair_dominance(pair_summary, highlight_pair=selected_pair, highlight_label=highlight_label)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        pair_passes = get_pair_pass_options(selected_pair)

        section_intro(
            "Shared pass explorer",
            "These are the passes where both defenders in the selected pair are present. Choose one to inspect frame-level PSV differences on the pitch.",
        )

        pass_display = pair_passes[
            ["match_id", "Pass_id", "pair_psv_diff", "max_pair_psv_diff", "p1_psv", "p2_psv", "n_frames"]
        ].rename(
            columns={
                "match_id": "Match",
                "Pass_id": "Pass",
                "pair_psv_diff": "Avg PSV diff",
                "max_pair_psv_diff": "Max PSV diff",
                "p1_psv": "Player 1 PSV",
                "p2_psv": "Player 2 PSV",
                "n_frames": "Frames",
            }
        )

        selected_pass_state = st.dataframe(
            prepare_table(
                pass_display,
                {
                    "Avg PSV diff": 3,
                    "Max PSV diff": 3,
                    "Player 1 PSV": 3,
                    "Player 2 PSV": 3,
                    "Frames": 0,
                },
            ),
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            height=300,
        )

        selected_pass_rows = selected_pass_state["selection"]["rows"]
        if len(selected_pass_rows) == 0:
            st.info("Select one pass from the table to view the frame-by-frame pitch snapshot.")
        else:
            selected_pass = pair_passes.iloc[selected_pass_rows]
            selected_match_id = selected_pass["match_id"].iloc[0]
            selected_pass_id = selected_pass["Pass_id"].iloc[0]
            selected_frames = get_pair_pass_frames(
                selected_pair=selected_pair,
                match_id=selected_match_id,
                pass_id=selected_pass_id,
            )

            frames = sorted(selected_frames["Frame"].tolist())
            if len(frames) == 1:
                frame_nr = frames[0]
                st.caption(f"Only one frame is available for this pass: {frame_nr}")
            else:
                frame_nr = st.select_slider("Frame", options=frames, value=frames[0], key="pair_frame_slider")

            selected_frame_row = selected_frames.loc[selected_frames["Frame"].eq(frame_nr)].iloc[0]
            match_id = int(selected_frame_row["match_id"])
            frame_psv_map = (
                selected_frames.loc[selected_frames["Frame"].eq(frame_nr), ["player_id", "PSV"]]
                .drop_duplicates(subset=["player_id"])
                .set_index("player_id")["PSV"]
                .to_dict()
            )

            top_cols = st.columns(4)
            top_cols[0].metric("Match", f"{match_id}")
            top_cols[1].metric("Pass ID", f"{int(selected_pass['Pass_id'].iloc[0])}")
            top_cols[2].metric("Frames in pass", f"{len(frames)}")
            top_cols[3].metric("Current frame", f"{frame_nr}")

            tracking, player_lookup, jersey_colors, number_colors, team_names = get_match_resources(match_id)
            fig, ax = plot_pair_frame(
                frame_idx=int(selected_frame_row["Frame"]),
                tracking=tracking,
                player_lookup=player_lookup,
                jersey_colors=jersey_colors,
                number_colors=number_colors,
                team_names=team_names,
                selected_pair=selected_pair,
                frame_psv_map=frame_psv_map,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
