import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --- ΡΥΘΜΙΣΕΙΣ ΕΦΑΡΜΟΓΗΣ ---
st.set_page_config(
    page_title="Ensemble Weather App", page_icon="🌊", layout="wide"
)

# --- ΚΥΡΙΩΣ ΕΦΑΡΜΟΓΗ ---
st.title("🌊 Πρόγνωση 14 Ημερών (Ensemble Consensus - 5 Μοντέλα)")
st.markdown(
    "Στατιστική σύνθεση από ECMWF, GFS, ICON, GEM & Meteo-France. "
    "Δείχνει **πότε συμφωνούν** τα μοντέλα και **πόσο αβέβαιη** "
    "είναι κάθε πρόβλεψη."
)

# --- SIDEBAR ---
st.sidebar.header("📍 Επιλογή Τοποθεσίας")

preset_locations = [
    "Σύμη", "Ρόδος", "Ντάτσα", "Φετιγιέ", "Αττάλεια",
    "Καστελόριζο", "Αθήνα", "Θεσσαλονίκη", "Πάτρα", "Ηράκλειο",
]

selection_mode = st.sidebar.radio(
    "Τρόπος επιλογής:",
    ["Γρήγορη Επιλογή (Top 10)", "Χειροκίνητη Πληκτρολόγηση"],
)

if selection_mode == "Γρήγορη Επιλογή (Top 10)":
    location_query = st.sidebar.selectbox("Επιλέξτε περιοχή:", preset_locations)
else:
    location_query = st.sidebar.text_input(
        "Γράψτε τοποθεσία (π.χ. Symi, Antalya):", value="Σύμη", max_chars=50
    )

# --- ΣΤΑΘΕΡΕΣ ---
MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_global,meteofrance_seamless"
MIN_MODELS = 3

PARAMS = {
    "pressure_msl":              {"label": "Πίεση (hPa)",      "scale": 6.0,  "weight": 0.20},
    "wind_speed_10m":            {"label": "Άνεμος (km/h)",     "scale": 12.0, "weight": 0.20},
    "wind_gusts_10m":            {"label": "Ριπές (km/h)",      "scale": 18.0, "weight": 0.15},
    "precipitation":             {"label": "Βροχή (mm)",        "scale": 3.0,  "weight": 0.15},
    "temperature_2m":            {"label": "Θερμ. (°C)",        "scale": 6.0,  "weight": 0.10},
    "wind_direction_10m":        {"label": "Διεύθυνση (°)",      "scale": 60.0, "weight": 0.10},
    "precipitation_probability": {"label": "Πιθαν. Βροχής (%)", "scale": 35.0, "weight": 0.05},
    "cloud_cover":               {"label": "Νέφωση (%)",        "scale": 45.0, "weight": 0.05},
    "relative_humidity_2m":      {"label": "Υγρασία (%)",       "scale": 25.0, "weight": 0.00},
}

GREEK_DAYS = {
    "Mon": "Δευ", "Tue": "Τρι", "Wed": "Τετ",
    "Thu": "Πεμ", "Fri": "Παρ", "Sat": "Σαβ", "Sun": "Κυρ",
}

COMPASS = ["Β", "ΒΑ", "Α", "ΝΑ", "Ν", "ΝΔ", "Δ", "ΒΔ"]


# --- ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ---
@st.cache_data(ttl=3600)
def get_coordinates(name):
    geo_url = (
        f"https://geocoding-api.open-meteo.com/v1/search?"
        f"name={name}&count=1&language=el&format=json"
    )
    try:
        res = requests.get(geo_url, timeout=10)
        data = res.json()
        if "results" in data and len(data["results"]) > 0:
            loc = data["results"][0]
            return (
                loc["latitude"], loc["longitude"],
                loc.get("name", name), loc.get("country", ""),
                loc.get("timezone", "UTC"),
            )
    except Exception as e:
        st.error(f"Σφάλμα geocoding: {e}")
    return None, None, None, None, None


def get_extended_fishing_weather(lat, lon, timezone):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,"
        f"precipitation_probability,pressure_msl,wind_speed_10m,"
        f"wind_gusts_10m,wind_direction_10m,cloud_cover&"
        f"models={MODELS}&forecast_days=14&timezone={timezone}"
    )
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Σφάλμα λήψης δεδομένων: {e}")
        return None


def circular_stats_deg(deg_df):
    """Κυκλικό median + circular std (σε μοίρες) ανά γραμμή."""
    rad = np.deg2rad(deg_df.astype(float))
    cos_mean = np.cos(rad).mean(axis=1, skipna=True)
    sin_mean = np.sin(rad).mean(axis=1, skipna=True)
    R = np.sqrt(cos_mean**2 + sin_mean**2).clip(lower=1e-9, upper=1.0)
    median_deg = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360) % 360
    std_deg = np.rad2deg(np.sqrt(-2 * np.log(R)))
    return median_deg, std_deg


def compass_gr(deg):
    if pd.isna(deg):
        return "—"
    idx = int((deg + 22.5) // 45) % 8
    return f"{COMPASS[idx]} {int(round(deg))}°"


def agreement_emoji(val):
    if pd.isna(val):
        return "—"
    v = int(round(val))
    if v >= 75:
        return f"🟢 {v}%"
    if v >= 50:
        return f"🟡 {v}%"
    return f"🔴 {v}%"


def fmt_time(dt):
    day = GREEK_DAYS.get(dt.strftime("%a"), dt.strftime("%a"))
    return f"{day} {dt.strftime('%d/%m %H:%M')}"


def fmt_pm(med, std):
    if pd.isna(med):
        return "—"
    if pd.isna(std):
        return f"{med:.1f}"
    return f"{med:.1f} ± {std:.1f}"


# --- ΚΥΡΙΩΣ ΡΟΗ ---
if location_query:
    with st.spinner(f"Λήψη δεδομένων για '{location_query}'..."):
        lat, lon, resolved_name, country, tz = get_coordinates(location_query)

        if lat is not None and lon is not None:
            st.success(
                f"📍 **{resolved_name}** ({country}) | "
                f"Συντεταγμένες: {lat:.3f}, {lon:.3f} | Ζώνη Ώρας: `{tz}`"
            )

            raw_data = get_extended_fishing_weather(lat, lon, tz)

            if raw_data and "hourly" in raw_data:
                df = pd.DataFrame(raw_data["hourly"])
                df["time"] = pd.to_datetime(df["time"])

                consensus = pd.DataFrame()
                consensus["Ώρα"] = df["time"]

                # Πλήθος διαθέσιμων μοντέλων ανά ώρα
                ref_cols = [c for c in df.columns if c.startswith("temperature_2m_")]
                consensus["n_models"] = (
                    df[ref_cols].notna().sum(axis=1) if ref_cols else 0
                )

                # Consensus + αβεβαιότητα ανά παράμετρο
                for param, meta in PARAMS.items():
                    model_cols = [c for c in df.columns if c.startswith(param + "_")]
                    if not model_cols:
                        continue

                    if param == "wind_direction_10m":
                        med, std = circular_stats_deg(df[model_cols])
                        consensus[f"{param}_median"] = med
                        consensus[f"{param}_std"] = std
                    else:
                        consensus[f"{param}_median"] = df[model_cols].median(axis=1)
                        consensus[f"{param}_std"] = df[model_cols].std(axis=1)

                    consensus[f"{param}_min"] = df[model_cols].min(axis=1)
                    consensus[f"{param}_max"] = df[model_cols].max(axis=1)

                    # Κανονικοποιημένη συμφωνία 0-1
                    agree = (1 - consensus[f"{param}_std"] / meta["scale"]).clip(0, 1)
                    consensus[f"{param}_agreement"] = agree

                # Συνολική σταθμισμένη συμφωνία 0-100
                w_sum = sum(m["weight"] for m in PARAMS.values() if m["weight"] > 0)
                overall = None
                for param, meta in PARAMS.items():
                    if meta["weight"] == 0:
                        continue
                    col = f"{param}_agreement"
                    if col not in consensus:
                        continue
                    contrib = consensus[col].fillna(0) * meta["weight"]
                    overall = contrib if overall is None else overall + contrib
                consensus["agreement"] = (
                    (overall / w_sum * 100).round(0) if overall is not None else np.nan
                )

                # Τάση πίεσης (3ωρη)
                consensus["pressure_trend"] = (
                    consensus["pressure_msl_median"]
                    - consensus["pressure_msl_median"].shift(3)
                ).fillna(0)

                # Φιλτράρισμα παρελθόντος + ελάχιστων μοντέλων
                try:
                    now = pd.Timestamp.now(tz=tz).tz_localize(None)
                except Exception:
                    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

                df_fc = consensus[consensus["Ώρα"] >= now].copy()
                df_fc = df_fc[df_fc["n_models"] >= MIN_MODELS].copy()

                if df_fc.empty:
                    st.warning("Δεν βρέθηκαν επαρκή δεδομένα για το μέλλον.")
                else:
                    # -------- ΜΕΤΡΙΚΕΣ ΚΟΡΥΦΗΣ --------
                    avg_agree = df_fc["agreement"].mean()
                    high_agree = int((df_fc["agreement"] >= 75).sum())
                    low_agree = int((df_fc["agreement"] < 50).sum())
                    total_hrs = len(df_fc)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Μέση συμφωνία", f"{avg_agree:.0f}%")
                    c2.metric("Ώρες υψηλής συμφωνίας 🟢", f"{high_agree}")
                    c3.metric("Ώρες χαμηλής συμφωνίας 🔴", f"{low_agree}")
                    c4.metric("Σύνολο ωρών", f"{total_hrs}")

                    # -------- SLIDER --------
                    slider_max = len(df_fc)
                    slider_default = min(72, slider_max)
                    hours_slider = st.slider(
                        "Ώρες προβολής:",
                        min_value=min(24, slider_max),
                        max_value=slider_max,
                        value=slider_default,
                        step=1,
                    )
                    view = df_fc.head(hours_slider)

                    # -------- TABS --------
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📋 Σύνοψη",
                        "📊 Αβεβαιότητα",
                        "🔬 Ανά Μοντέλο",
                        "📅 Ημερήσια",
                    ])

                    # --- TAB 1: Σύνοψη ---
                    with tab1:
                        st.caption(
                            "Σταθμισμένος median. Η στήλη **Συμφωνία** δείχνει "
                            "πόσο κοντά είναι τα 5 μοντέλα μεταξύ τους."
                        )
                        summary = pd.DataFrame({
                            "Ώρα": view["Ώρα"].apply(fmt_time),
                            "Πίεση": view["pressure_msl_median"].round(1),
                            "Τάση": view["pressure_trend"].round(1),
                            "Άνεμος": view["wind_speed_10m_median"].round(1),
                            "Ριπές": view["wind_gusts_10m_median"].round(1),
                            "Δ/νση": view["wind_direction_10m_median"].apply(compass_gr),
                            "Βροχή": view["precipitation_median"].round(2),
                            "Πιθαν.%": view["precipitation_probability_median"].round(0).astype("Int64"),
                            "Νέφωση": view["cloud_cover_median"].round(0).astype("Int64"),
                            "Θερμ.": view["temperature_2m_median"].round(1),
                            "Συμφωνία": view["agreement"].apply(agreement_emoji),
                            "Μοντ.": view["n_models"].astype(int),
                        })
                        st.dataframe(summary, use_container_width=True, hide_index=True)

                    # --- TAB 2: Αβεβαιότητα ---
                    with tab2:
                        st.caption(
                            "Κάθε κελί: **median ± std** μεταξύ των 5 μοντέλων. "
                            "Μεγάλο ± σημαίνει μεγάλη διαφωνία/αβεβαιότητα."
                        )
                        unc = pd.DataFrame({
                            "Ώρα": view["Ώρα"].apply(fmt_time),
                            "Πίεση": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["pressure_msl_median"],
                                    view["pressure_msl_std"],
                                )
                            ],
                            "Άνεμος": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["wind_speed_10m_median"],
                                    view["wind_speed_10m_std"],
                                )
                            ],
                            "Ριπές": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["wind_gusts_10m_median"],
                                    view["wind_gusts_10m_std"],
                                )
                            ],
                            "Δ/νση": [
                                f"{compass_gr(m)} ±{s:.0f}°"
                                if not pd.isna(s) else compass_gr(m)
                                for m, s in zip(
                                    view["wind_direction_10m_median"],
                                    view["wind_direction_10m_std"],
                                )
                            ],
                            "Βροχή": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["precipitation_median"],
                                    view["precipitation_std"],
                                )
                            ],
                            "Πιθαν.%": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["precipitation_probability_median"],
                                    view["precipitation_probability_std"],
                                )
                            ],
                            "Νέφωση": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["cloud_cover_median"],
                                    view["cloud_cover_std"],
                                )
                            ],
                            "Θερμ.": [
                                fmt_pm(m, s)
                                for m, s in zip(
                                    view["temperature_2m_median"],
                                    view["temperature_2m_std"],
                                )
                            ],
                            "Συμφωνία": view["agreement"].apply(agreement_emoji),
                        })
                        st.dataframe(unc, use_container_width=True, hide_index=True)

                    # --- TAB 3: Ανά Μοντέλο ---
                    with tab3:
                        st.caption(
                            "Δες τις ωμές τιμές κάθε μοντέλου για μια "
                            "παράμετρο — χρήσιμο για να εντοπίσεις τον "
                            "«αιρετικό»."
                        )
                        param_choice = st.selectbox(
                            "Παράμετρος:",
                            options=list(PARAMS.keys()),
                            format_func=lambda p: PARAMS[p]["label"],
                        )
                        model_cols = [
                            c for c in df.columns
                            if c.startswith(param_choice + "_")
                        ]
                        model_cols = sorted(model_cols)
                        if model_cols:
                            per_model = pd.DataFrame({
                                "Ώρα": view["Ώρα"].apply(fmt_time).values,
                            })
                            raw = df.loc[view.index, ["time"] + model_cols].copy()
                            for mc in model_cols:
                                short = (
                                    mc.replace(param_choice + "_", "")
                                    .replace("_seamless", "")
                                    .replace("_ifs025", "")
                                    .replace("_global", "")
                                )
                                per_model[short] = raw[mc].values
                            per_model["Διαφορά"] = (
                                view[f"{param_choice}_max"]
                                - view[f"{param_choice}_min"]
                            ).round(2).values
                            per_model["Συμφωνία"] = (
                                view["agreement"].apply(agreement_emoji).values
                            )
                            st.dataframe(
                                per_model,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("Δεν υπάρχουν δεδομένα για αυτή την παράμετρο.")

                    # --- TAB 4: Ημερήσια ---
                    with tab4:
                        st.caption(
                            "Σύνοψη ανά ημέρα. Η **Συμφωνία** είναι ο μέσος "
                            "όρος των ωριαίων τιμών· η **Χειρότερη** δείχνει "
                            "την πιο αβέβαιη ώρα της ημέρας."
                        )
                        daily = view.copy()
                        daily["Ημέρα"] = daily["Ώρα"].dt.date

                        def daily_stats(g):
                            return pd.Series({
                                "Θερμ. min": g["temperature_2m_median"].min(),
                                "Θερμ. max": g["temperature_2m_median"].max(),
                                "Άνεμος max": g["wind_speed_10m_median"].max(),
                                "Ριπές max": g["wind_gusts_10m_median"].max(),
                                "Βροχή mm": g["precipitation_median"].sum(),
                                "Πιθαν.% max": g["precipitation_probability_median"].max(),
                                "Συμφωνία": g["agreement"].mean(),
                                "Χειρότερη": g["agreement"].min(),
                                "Ώρες": len(g),
                            })

                        day_df = (
                            daily.groupby("Ημέρα").apply(daily_stats).reset_index()
                        )
                        day_df["Ημέρα"] = day_df["Ημέρα"].apply(
                            lambda d: (
                                f"{GREEK_DAYS.get(d.strftime('%a'), d.strftime('%a'))}"
                                f" {d.strftime('%d/%m')}"
                            )
                        )
                        for col in ["Θερμ. min", "Θερμ. max", "Άνεμος max",
                                    "Ριπές max", "Βροχή mm", "Πιθαν.% max"]:
                            day_df[col] = day_df[col].round(1)
                        day_df["Συμφωνία"] = day_df["Συμφωνία"].apply(agreement_emoji)
                        day_df["Χειρότερη"] = day_df["Χειρότερη"].apply(agreement_emoji)

                        st.dataframe(
                            day_df, use_container_width=True, hide_index=True
                        )

                    # -------- CSV --------
                    csv = df_fc.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label=f"📥 Λήψη CSV 14 ημερών για {resolved_name}",
                        data=csv,
                        file_name=(
                            f"{resolved_name.lower()}_ensemble_14d_"
                            f"{datetime.date.today().strftime('%Y%m%d')}.csv"
                        ),
                        mime="text/csv",
                    )
            else:
                st.error("Αποτυχία λήψης μετεωρολογικών δεδομένων.")
        else:
            st.warning(
                "⚠️ Η τοποθεσία δεν βρέθηκε. Δοκίμασε λατινικά "
                "(π.χ. Antalya, Symi, Rhodes)."
            )
