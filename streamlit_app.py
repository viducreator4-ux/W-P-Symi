import datetime
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --- ΡΥΘΜΙΣΕΙΣ ΕΦΑΡΜΟΓΗΣ ---
st.set_page_config(
    page_title="Ensemble Weather App", page_icon="🌊", layout="wide"
)

# --- ΕΛΕΓΧΟΣ ΚΩΔΙΚΟΥ ΠΡΟΣΒΑΣΗΣ ---
def check_password():
    if "APP_PASSWORD" not in st.secrets:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 Ψαρευτική Πρόγνωση")
    st.markdown("Βάλε τον κωδικό για να δεις την πρόγνωση.")

    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Κωδικός:", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Είσοδος", type="primary")

    if submitted:
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Λάθος κωδικός. Δοκίμασε ξανά.")
    return False


if not check_password():
    st.stop()
# ----------------------------------

# --- ΚΥΡΙΩΣ ΕΦΑΡΜΟΓΗ ---
st.title("🌊 Πρόγνωση 14 Ημερών (Ensemble Consensus - 5 Μοντέλα)")
st.markdown(
    "Στατιστική σύνθεση από ECMWF, GFS, ICON, GEM & Meteo-France. "
    "Η στήλη **Συμφωνία** δείχνει πόσο κοντά είναι τα 5 μοντέλα — "
    "βασισμένη σε **robust** εκτίμηση (MAD) που αγνοεί outliers."
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

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Αποσύνδεση"):
    st.session_state["authenticated"] = False
    st.rerun()

# --- ΣΤΑΘΕΡΕΣ ---
MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_global,meteofrance_seamless"
MIN_MODELS = 3

# Το "scale" = η τυπική απόκλιση στην οποία θεωρούμε ότι τα μοντέλα
# διαφωνούν πλήρως (agreement = 0%). Προέρχεται από φυσικά όρια:
#   - Πίεση: 6 hPa = μεγάλη απόκλιση για synoptic κλίμακα
#   - Άνεμος: 12 km/h ≈ 6.5 knots
#   - Ριπές: 18 km/h
#   - Βροχή: 3 mm σε 1 ώρα = σοβαρή διαφωνία
#   - Θερμ.: 6°C
#   - Διεύθυνση: 60° (κυκλικό)
PARAMS = {
    "pressure_msl":              {"label": "Πίεση (hPa)",      "scale": 6.0,  "weight": 0.20, "important": True},
    "wind_speed_10m":            {"label": "Άνεμος (km/h)",     "scale": 12.0, "weight": 0.20, "important": True},
    "wind_gusts_10m":            {"label": "Ριπές (km/h)",      "scale": 18.0, "weight": 0.15, "important": True},
    "precipitation":             {"label": "Βροχή (mm)",        "scale": 3.0,  "weight": 0.15, "important": True},
    "temperature_2m":            {"label": "Θερμ. (°C)",        "scale": 6.0,  "weight": 0.10, "important": False},
    "wind_direction_10m":        {"label": "Διεύθυνση (°)",      "scale": 60.0, "weight": 0.10, "important": False},
    "precipitation_probability": {"label": "Πιθαν. Βροχής (%)", "scale": 35.0, "weight": 0.05, "important": False},
    "cloud_cover":               {"label": "Νέφωση (%)",        "scale": 45.0, "weight": 0.05, "important": False},
    "relative_humidity_2m":      {"label": "Υγρασία (%)",       "scale": 25.0, "weight": 0.00, "important": False},
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


def fetch_with_retry(url, max_retries=3, backoff=1.5, timeout=20):
    """HTTP GET με retry + exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
    st.error(f"Σφάλμα λήψης δεδομένων μετά από {max_retries} προσπάθειες: {last_error}")
    return None


def get_extended_fishing_weather(lat, lon, timezone):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,"
        f"precipitation_probability,pressure_msl,wind_speed_10m,"
        f"wind_gusts_10m,wind_direction_10m,cloud_cover&"
        f"models={MODELS}&forecast_days=14&timezone={timezone}"
    )
    return fetch_with_retry(url)


def circular_mean_std(deg_df):
    """
    Κυκλικός μέσος όρος + circular std (σε μοίρες).
    ΣΗΜΕΙΩΣΗ: Είναι MEAN, όχι median — ο median δεν ορίζεται μονοσήμαντα
    στον κύκλο. Ο circular mean είναι ο καθιερωμένος estimator στη μετεωρολογία.
    """
    rad = np.deg2rad(deg_df.astype(float))
    cos_mean = np.cos(rad).mean(axis=1, skipna=True)
    sin_mean = np.sin(rad).mean(axis=1, skipna=True)
    R = np.sqrt(cos_mean**2 + sin_mean**2).clip(lower=1e-9, upper=1.0)
    mean_deg = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360) % 360
    # circular std: sqrt(-2 * ln(R)) σε radians, μετατροπή σε μοίρες
    std_deg = np.rad2deg(np.sqrt(-2 * np.log(R)))
    return mean_deg, std_deg


def robust_spread(df_vals, median_vals):
    """
    Robust εκτίμηση διασποράς βασισμένη σε MAD.
    MAD = median(|x_i - median(x)|)
    Robust sigma ≈ 1.4826 * MAD (για κανονική κατανομή).
    Ανθεκτικό σε outliers, σε αντίθεση με το std.
    """
    mad = df_vals.sub(median_vals, axis=0).abs().median(axis=1)
    return 1.4826 * mad


def compass_gr(deg):
    if pd.isna(deg):
        return "—"
    idx = int((deg + 22.5) // 45) % 8
    return f"{COMPASS[idx]} {int(round(deg))}°"


def agreement_pct(spread, scale):
    """Γραμμική μετατροπή spread → % συμφωνίας [0, 100]."""
    return (100 * (1 - spread / scale)).clip(lower=0, upper=100)


def agreement_emoji(pct):
    if pd.isna(pct):
        return "—"
    v = int(round(pct))
    if v >= 75:
        return f"🟢 {v}%"
    if v >= 50:
        return f"🟡 {v}%"
    return f"🔴 {v}%"


def fmt_time(dt):
    day = GREEK_DAYS.get(dt.strftime("%a"), dt.strftime("%a"))
    return f"{day} {dt.strftime('%d/%m %H:%M')}"


def fmt_pm(med, spread):
    """Median ± robust spread."""
    if pd.isna(med):
        return "—"
    if pd.isna(spread) or spread < 0.05:
        return f"{med:.1f}"
    return f"{med:.1f} ± {spread:.1f}"


def fmt_range(min_v, max_v, decimals=1):
    """Εύρος [min, max]."""
    if pd.isna(min_v) or pd.isna(max_v):
        return "—"
    return f"[{min_v:.{decimals}f}, {max_v:.{decimals}f}]"


def angular_diff(a, b):
    """Ελάχιστη κυκλική απόσταση σε μοίρες (0-180)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def angular_diff_vec(a_series, b_series):
    """Διανυσματική εκδοχή του angular_diff."""
    d = (a_series - b_series).abs() % 360
    return np.minimum(d, 360 - d)


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

                # Ποιες παράμετροι είναι διαθέσιμες (έχουν στήλες μοντέλων)
                available_params = {
                    p: m for p, m in PARAMS.items()
                    if any(c.startswith(p + "_") for c in df.columns)
                }
                if not available_params:
                    st.error("Δεν βρέθηκαν διαθέσιμες παράμετροι στα δεδομένα.")
                    st.stop()

                consensus = pd.DataFrame()
                consensus["Ώρα"] = df["time"]

                # Πλήθος διαθέσιμων μοντέλων ΑΝΑ ΠΑΡΑΜΕΤΡΟ
                for param in available_params:
                    model_cols = [
                        c for c in df.columns if c.startswith(param + "_")
                    ]
                    consensus[f"{param}_n"] = df[model_cols].notna().sum(axis=1)

                # Πλήθος ελάχιστου across των "σημαντικών" παραμέτρων
                important_params = [
                    p for p, m in available_params.items() if m["important"]
                ]
                if important_params:
                    n_cols = [f"{p}_n" for p in important_params]
                    consensus["n_models_min"] = consensus[n_cols].min(axis=1)
                else:
                    all_n_cols = [f"{p}_n" for p in available_params]
                    consensus["n_models_min"] = consensus[all_n_cols].min(axis=1)

                # Consensus + spread ανά παράμετρο
                for param, meta in available_params.items():
                    model_cols = [
                        c for c in df.columns if c.startswith(param + "_")
                    ]
                    if not model_cols:
                        continue

                    if param == "wind_direction_10m":
                        mean_deg, std_deg = circular_mean_std(df[model_cols])
                        consensus[f"{param}_median"] = mean_deg
                        consensus[f"{param}_spread"] = std_deg
                    else:
                        median_v = df[model_cols].median(axis=1)
                        consensus[f"{param}_median"] = median_v
                        # Robust spread (MAD-based) — ανθεκτικό σε outliers
                        consensus[f"{param}_spread"] = robust_spread(
                            df[model_cols], median_v
                        )

                    # Εύρος [min, max] για διαφάνεια
                    consensus[f"{param}_min"] = df[model_cols].min(axis=1)
                    consensus[f"{param}_max"] = df[model_cols].max(axis=1)

                    # Κανονικοποιημένη συμφωνία 0-100
                    consensus[f"{param}_agreement"] = agreement_pct(
                        consensus[f"{param}_spread"], meta["scale"]
                    )

                # Συνολική σταθμισμένη συμφωνία
                w_sum = sum(
                    m["weight"] for m in available_params.values()
                    if m["weight"] > 0
                )
                overall = None
                for param, meta in available_params.items():
                    if meta["weight"] == 0:
                        continue
                    col = f"{param}_agreement"
                    if col not in consensus:
                        continue
                    contrib = consensus[col].fillna(0) * meta["weight"]
                    overall = contrib if overall is None else overall + contrib
                consensus["agreement"] = (
                    (overall / w_sum).round(0) if overall is not None else np.nan
                )

                # Τάση πίεσης — ΧΡΟΝΙΚΑ βασισμένη (3 ώρες), όχι 3 γραμμές
                if "pressure_msl_median" in consensus.columns:
                    consensus = consensus.sort_values("Ώρα").reset_index(drop=True)
                    pressure_s = consensus.set_index("Ώρα")["pressure_msl_median"]
                    trend_s = pressure_s - pressure_s.shift(freq="3h")
                    consensus["pressure_trend"] = (
                        trend_s.values.round(2)
                    )
                    consensus["pressure_trend"] = (
                        consensus["pressure_trend"].fillna(0)
                    )
                else:
                    consensus["pressure_trend"] = 0.0

                # Φιλτράρισμα παρελθόντος + ελάχιστων μοντέλων
                try:
                    now = pd.Timestamp.now(tz=tz).tz_localize(None)
                except Exception:
                    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

                df_fc = consensus[consensus["Ώρα"] >= now].copy()
                df_fc = df_fc[df_fc["n_models_min"] >= MIN_MODELS].copy()

                if df_fc.empty:
                    st.warning(
                        "Δεν βρέθηκαν μελλοντικά δεδομένα με επαρκή αριθμό "
                        "μοντέλων (≥3)."
                    )
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

                    st.caption(
                        "ℹ️ *Η συμφωνία υπολογίζεται από **MAD** (robust "
                        "spread) και όχι από τυπική απόκλιση — αγνοεί μεμονωμένα "
                        "μοντέλα που ξεφεύγουν.*"
                    )

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
                            "Σταθμισμένος **robust median** ανά παράμετρο. "
                            "Η στήλη **Συμφωνία** δείχνει πόσο κοντά είναι τα "
                            "5 μοντέλα (🟢 ≥75%, 🟡 50-74%, 🔴 <50%)."
                        )
                        summary_cols = {
                            "Ώρα": view["Ώρα"].apply(fmt_time),
                        }
                        if "pressure_msl_median" in view.columns:
                            summary_cols["Πίεση"] = view["pressure_msl_median"].round(1)
                            summary_cols["Τάση/3h"] = view["pressure_trend"].round(2)
                        if "wind_speed_10m_median" in view.columns:
                            summary_cols["Άνεμος"] = view["wind_speed_10m_median"].round(1)
                        if "wind_gusts_10m_median" in view.columns:
                            summary_cols["Ριπές"] = view["wind_gusts_10m_median"].round(1)
                        if "wind_direction_10m_median" in view.columns:
                            summary_cols["Δ/νση"] = (
                                view["wind_direction_10m_median"].apply(compass_gr)
                            )
                        if "precipitation_median" in view.columns:
                            summary_cols["Βροχή"] = view["precipitation_median"].round(2)
                        if "precipitation_probability_median" in view.columns:
                            summary_cols["Πιθαν.%"] = (
                                view["precipitation_probability_median"]
                                .round(0).astype("Int64")
                            )
                        if "cloud_cover_median" in view.columns:
                            summary_cols["Νέφωση"] = (
                                view["cloud_cover_median"].round(0).astype("Int64")
                            )
                        if "temperature_2m_median" in view.columns:
                            summary_cols["Θερμ."] = view["temperature_2m_median"].round(1)
                        summary_cols["Συμφωνία"] = view["agreement"].apply(agreement_emoji)
                        summary_cols["Μοντ."] = view["n_models_min"].astype(int)

                        st.dataframe(
                            pd.DataFrame(summary_cols),
                            use_container_width=True,
                            hide_index=True,
                        )

                    # --- TAB 2: Αβεβαιότητα ---
                    with tab2:
                        st.caption(
                            "Κάθε κελί: **median ± robust spread (MAD)** "
                            "και **[min, max]** μεταξύ των 5 μοντέλων. "
                            "Μεγάλο spread ή ευρύ [min, max] = μεγάλη "
                            "διαφωνία."
                        )
                        unc_cols = {"Ώρα": view["Ώρα"].apply(fmt_time)}

                        def make_unc_cols(v, prefix, label, decimals=1):
                            if f"{prefix}_median" not in v.columns:
                                return None
                            med = v[f"{prefix}_median"]
                            spread = v[f"{prefix}_spread"]
                            mn = v[f"{prefix}_min"]
                            mx = v[f"{prefix}_max"]
                            return [
                                f"{fmt_pm(m, s)}  {fmt_range(lo, hi, decimals)}"
                                for m, s, lo, hi in zip(med, spread, mn, mx)
                            ]

                        if "pressure_msl_median" in view.columns:
                            unc_cols["Πίεση"] = make_unc_cols(
                                view, "pressure_msl", "Πίεση", 1
                            )
                        if "wind_speed_10m_median" in view.columns:
                            unc_cols["Άνεμος"] = make_unc_cols(
                                view, "wind_speed_10m", "Άνεμος", 1
                            )
                        if "wind_gusts_10m_median" in view.columns:
                            unc_cols["Ριπές"] = make_unc_cols(
                                view, "wind_gusts_10m", "Ριπές", 1
                            )
                        if "precipitation_median" in view.columns:
                            unc_cols["Βροχή"] = make_unc_cols(
                                view, "precipitation", "Βροχή", 2
                            )
                        if "temperature_2m_median" in view.columns:
                            unc_cols["Θερμ."] = make_unc_cols(
                                view, "temperature_2m", "Θερμ.", 1
                            )
                        if "wind_direction_10m_median" in view.columns:
                            unc_cols["Δ/νση"] = [
                                f"{compass_gr(m)} ±{s:.0f}°"
                                if not pd.isna(s) else compass_gr(m)
                                for m, s in zip(
                                    view["wind_direction_10m_median"],
                                    view["wind_direction_10m_spread"],
                                )
                            ]
                        unc_cols["Συμφωνία"] = view["agreement"].apply(agreement_emoji)

                        st.dataframe(
                            pd.DataFrame(unc_cols),
                            use_container_width=True,
                            hide_index=True,
                        )

                    # --- TAB 3: Ανά Μοντέλο ---
                    with tab3:
                        st.caption(
                            "Δες τις ωμές τιμές κάθε μοντέλου για μια "
                            "παράμετρο — χρήσιμο για εντοπισμό του "
                            "«αιρετικού» μοντέλου."
                        )
                        param_choice = st.selectbox(
                            "Παράμετρος:",
                            options=list(available_params.keys()),
                            format_func=lambda p: available_params[p]["label"],
                        )
                        model_cols = sorted([
                            c for c in df.columns
                            if c.startswith(param_choice + "_")
                        ])
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

                            if param_choice == "wind_direction_10m":
                                # Κυκλική διαφορά (σωστή!)
                                diff = angular_diff_vec(
                                    view[f"{param_choice}_max"],
                                    view[f"{param_choice}_min"],
                                )
                                per_model["Κυκλ. Διαφ."] = diff.round(1).values
                            else:
                                per_model["Spread"] = (
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
                            "την πιο αβέβαιη ώρα."
                        )
                        daily = view.copy()
                        daily["Ημέρα"] = daily["Ώρα"].dt.date

                        def daily_stats(g):
                            s = {
                                "Ώρες": len(g),
                                "Συμφωνία": g["agreement"].mean(),
                                "Χειρότερη": g["agreement"].min(),
                            }
                            if "temperature_2m_median" in g.columns:
                                s["Θερμ. min"] = g["temperature_2m_median"].min()
                                s["Θερμ. max"] = g["temperature_2m_median"].max()
                            if "wind_speed_10m_median" in g.columns:
                                s["Άνεμος max"] = g["wind_speed_10m_median"].max()
                            if "wind_gusts_10m_median" in g.columns:
                                s["Ριπές max"] = g["wind_gusts_10m_median"].max()
                            if "precipitation_median" in g.columns:
                                s["Βροχή mm"] = g["precipitation_median"].sum()
                            if "precipitation_probability_median" in g.columns:
                                s["Πιθαν.% max"] = g[
                                    "precipitation_probability_median"
                                ].max()
                            return pd.Series(s)

                        day_df = (
                            daily.groupby("Ημέρα").apply(daily_stats).reset_index()
                        )
                        day_df["Ημέρα"] = day_df["Ημέρα"].apply(
                            lambda d: (
                                f"{GREEK_DAYS.get(d.strftime('%a'), d.strftime('%a'))}"
                                f" {d.strftime('%d/%m')}"
                            )
                        )
                        for col in [
                            "Θερμ. min", "Θερμ. max", "Άνεμος max",
                            "Ριπές max", "Βροχή mm", "Πιθαν.% max",
                        ]:
                            if col in day_df.columns:
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
