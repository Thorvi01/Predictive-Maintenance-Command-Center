# app.py
# Streamlit dashboard — Fleet Health Monitor + Maintenance Copilot
# Run with: streamlit run app.py
from calibrate import load_temperature
T = load_temperature()   # loads the fitted scalar, returns 1.0 if not found
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

from dataset import get_dataloaders
from model import RULPredictor, mc_predict
from rag.retriever import MaintenanceRetriever

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FleetGuard AI — Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #E0E0E0; }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .critical { border-left: 4px solid #FF4444; }
    .high     { border-left: 4px solid #FF8C00; }
    .elevated { border-left: 4px solid #FFD700; }
    .healthy  { border-left: 4px solid #00E676; }
    .stButton>button {
        background-color: #1a1a2e;
        color: #00C4FF;
        border: 1px solid #00C4FF;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00C4FF;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ────────────────────
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        'models/rul_predictor.pt',
        map_location='cpu',
        weights_only=False
    )
    config = checkpoint['config']
    model  = RULPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


@st.cache_resource
def load_retriever():
    return MaintenanceRetriever()


@st.cache_data
def load_fleet_data():
    """Runs MC inference on all 100 test engines. Cached after first run."""
    model, config = load_model()
    _, _, test_loader = get_dataloaders(
        batch_size=100,
        val_split=config['val_split'],
        window_size=config['window_size']
    )
    X_test = torch.cat([X for X, _ in test_loader])
    y_test = torch.cat([y for _, y in test_loader]).numpy()

    mean_preds, std_preds, lower, upper, _ = mc_predict(
        model, X_test, n_samples=100, device='cpu'
    )

    # Apply temperature scaling for calibrated uncertainty
    std_preds = std_preds * T
    lower     = mean_preds - 1.645 * std_preds
    upper     = mean_preds + 1.645 * std_preds

    mean_preds = np.clip(mean_preds, 0, None)
    lower      = np.clip(lower, 0, None)

    fleet = []
    for i in range(len(X_test)):
        mean = float(mean_preds[i])
        lo   = float(lower[i])
        effective = min(mean, lo)

        if effective < 10:   risk, icon = "CRITICAL", "🔴"
        elif effective < 20: risk, icon = "HIGH",     "🟠"
        elif effective < 50: risk, icon = "ELEVATED", "🟡"
        elif effective < 90: risk, icon = "MODERATE", "🟢"
        else:                risk, icon = "HEALTHY",  "✅"

        fleet.append({
            'engine_id': i + 1,
            'true_rul':  float(y_test[i]),
            'mean_rul':  mean,
            'std':       float(std_preds[i]),
            'lower_ci':  float(lo),
            'upper_ci':  float(upper[i]),
            'ci_width':  float(upper[i] - lower[i]),
            'risk':      risk,
            'icon':      icon
        })

    return sorted(fleet, key=lambda x: min(x['mean_rul'], x['lower_ci']))


# ── Risk color map ───────────────────────────────────────────────
RISK_COLORS = {
    'CRITICAL': '#FF4444',
    'HIGH':     '#FF8C00',
    'ELEVATED': '#FFD700',
    'MODERATE': '#00C4FF',
    'HEALTHY':  '#00E676'
}


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ FleetGuard AI")
    st.markdown("*Predictive Maintenance Command Center*")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "", ["Fleet Overview", "Engine Deep Dive", "Maintenance Copilot",
             "Drift Monitor", "Model Performance"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### Model Info")
    st.markdown("**Architecture:** 2-layer LSTM")
    st.markdown("**Uncertainty:** MC Dropout (N=100)")
    st.markdown("**Dataset:** NASA C-MAPSS FD001")
    st.markdown("**Test RMSE:** 13.56 cycles")
    st.markdown("**ECE Score:** 0.2020")


# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading fleet data..."):
    fleet = load_fleet_data()
    retriever = load_retriever()

df = pd.DataFrame(fleet)

# ── Risk counts ──────────────────────────────────────────────────
risk_counts = df['risk'].value_counts()


# ════════════════════════════════════════════════════════════════
# PAGE 1: Fleet Overview
# ════════════════════════════════════════════════════════════════
if page == "Fleet Overview":
    st.markdown("# 🛩️ Fleet Health Overview")
    st.markdown("*Real-time RUL predictions with MC Dropout uncertainty — "
                "NASA C-MAPSS FD001*")
    st.divider()

    # ── KPI cards ──
    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, "🔴 Critical",  risk_counts.get('CRITICAL', 0),  "#FF4444"),
        (col2, "🟠 High",      risk_counts.get('HIGH', 0),      "#FF8C00"),
        (col3, "🟡 Elevated",  risk_counts.get('ELEVATED', 0),  "#FFD700"),
        (col4, "🟢 Moderate",  risk_counts.get('MODERATE', 0),  "#00C4FF"),
        (col5, "✅ Healthy",   risk_counts.get('HEALTHY', 0),   "#00E676"),
    ]
    for col, label, count, color in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<h2 style="color:{color};margin:0">{count}</h2>'
                f'<p style="margin:0;color:#aaa">{label}</p>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("")

    # ── RUL bar chart with CI ──
    st.markdown("### Fleet RUL Predictions with 90% Confidence Intervals")

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    engines   = [e['engine_id'] for e in fleet]
    means     = [e['mean_rul']  for e in fleet]
    lowers    = [e['lower_ci']  for e in fleet]
    uppers    = [e['upper_ci']  for e in fleet]
    colors    = [RISK_COLORS[e['risk']] for e in fleet]

    yerr_lower = [m - l for m, l in zip(means, lowers)]
    yerr_upper = [u - m for u, m in zip(uppers, means)]

    bars = ax.bar(range(len(engines)), means, color=colors,
                  alpha=0.8, width=0.7)
    ax.errorbar(range(len(engines)), means,
                yerr=[yerr_lower, yerr_upper],
                fmt='none', color='white', alpha=0.3,
                capsize=2, linewidth=0.8)

    ax.axhline(y=20, color='#FF4444', linestyle='--',
               linewidth=1, alpha=0.7, label='Critical threshold (20)')
    ax.axhline(y=50, color='#FFD700', linestyle='--',
               linewidth=1, alpha=0.5, label='Elevated threshold (50)')

    ax.set_xlabel('Engine (sorted by RUL)', color='#aaa')
    ax.set_ylabel('RUL (cycles)', color='#aaa')
    ax.set_title('Fleet RUL — Sorted Most Critical First',
                 color='white', fontsize=13)
    ax.tick_params(colors='#aaa')
    ax.set_xticks([])
    ax.legend(facecolor='#222', labelcolor='white', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    st.pyplot(fig)
    plt.close()

    # ── Fleet table ──
    st.markdown("### Fleet Status Table")
    display_df = df[['engine_id', 'true_rul', 'mean_rul',
                      'lower_ci', 'upper_ci', 'std', 'risk']].copy()
    display_df.columns = ['Engine', 'True RUL', 'Mean RUL',
                           'CI Lower', 'CI Upper', 'Uncertainty', 'Risk']
    display_df = display_df.round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=300,
        hide_index=True
    )


# ════════════════════════════════════════════════════════════════
# PAGE 2: Engine Deep Dive
# ════════════════════════════════════════════════════════════════
elif page == "Engine Deep Dive":
    st.markdown("# 🔍 Engine Deep Dive")
    st.divider()

    engine_ids = [e['engine_id'] for e in fleet]
    selected   = st.selectbox(
        "Select Engine",
        engine_ids,
        format_func=lambda x: f"Engine {x} — "
                               f"{next(e['icon'] for e in fleet if e['engine_id']==x)} "
                               f"{next(e['risk'] for e in fleet if e['engine_id']==x)}"
    )

    eng = next(e for e in fleet if e['engine_id'] == selected)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean RUL",      f"{eng['mean_rul']:.1f} cycles")
    col2.metric("True RUL",      f"{eng['true_rul']:.0f} cycles")
    col3.metric("90% CI",        f"[{eng['lower_ci']:.1f}, {eng['upper_ci']:.1f}]")
    col4.metric("Uncertainty",   f"±{eng['std']:.1f} cycles")

    risk_color = RISK_COLORS[eng['risk']]
    st.markdown(
        f'<div style="background:{risk_color}22;border-left:4px solid '
        f'{risk_color};padding:12px;border-radius:8px;margin:12px 0">'
        f'<b style="color:{risk_color}">{eng["icon"]} {eng["risk"]} RISK</b> — '
        f'Effective RUL (using lower CI bound): '
        f'{min(eng["mean_rul"], eng["lower_ci"]):.1f} cycles'
        f'</div>', unsafe_allow_html=True
    )

    # MC distribution plot
    st.markdown("### MC Dropout Prediction Distribution")
    model, config = load_model()
    _, _, test_loader = get_dataloaders(
        batch_size=100, val_split=0.2, window_size=30
    )
    X_test   = torch.cat([X for X, _ in test_loader])
    idx      = selected - 1
    mean_app, std_app, lower_app, upper_app, all_preds = mc_predict(
        model, X_test[idx:idx+1], n_samples=200, device='cpu'
    )

    # Apply temperature scaling for calibrated uncertainty
    std_app   = std_app * T
    lower_app = mean_app - 1.645 * std_app
    upper_app = mean_app + 1.645 * std_app

    samples = all_preds[:, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    ax.hist(samples, bins=30, color='#00C4FF', alpha=0.7,
            edgecolor='#003344')
    ax.axvline(eng['mean_rul'],  color='#FFCC00', linewidth=2,
               label=f"Mean = {eng['mean_rul']:.1f}")
    ax.axvline(eng['lower_ci'],  color='#FF4444', linewidth=1.5,
               linestyle='--', label=f"Lower CI = {eng['lower_ci']:.1f}")
    ax.axvline(eng['upper_ci'],  color='#00E676', linewidth=1.5,
               linestyle='--', label=f"Upper CI = {eng['upper_ci']:.1f}")
    ax.axvline(eng['true_rul'],  color='white',   linewidth=2,
               linestyle=':',  label=f"True RUL = {eng['true_rul']:.0f}")
    ax.set_xlabel('Predicted RUL (cycles)', color='#aaa')
    ax.set_ylabel('Frequency', color='#aaa')
    ax.set_title(f'200 MC Dropout Samples — Engine {selected}',
                 color='white')
    ax.tick_params(colors='#aaa')
    ax.legend(facecolor='#222', labelcolor='white', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════
# PAGE 3: Maintenance Copilot
# ════════════════════════════════════════════════════════════════
elif page == "Maintenance Copilot":
    st.markdown("# 🤖 Maintenance Copilot")
    st.markdown("*RAG-powered recommendations grounded in NASA documentation*")
    st.divider()

    engine_ids = [e['engine_id'] for e in fleet]
    selected   = st.selectbox(
        "Select Engine to Analyze",
        engine_ids,
        format_func=lambda x: f"Engine {x} — "
                               f"{next(e['icon'] for e in fleet if e['engine_id']==x)} "
                               f"{next(e['risk'] for e in fleet if e['engine_id']==x)}"
    )
    eng = next(e for e in fleet if e['engine_id'] == selected)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean RUL",    f"{eng['mean_rul']:.1f} cycles")
    col2.metric("90% CI",      f"[{eng['lower_ci']:.1f}, {eng['upper_ci']:.1f}]")
    col3.metric("Risk Level",  f"{eng['icon']} {eng['risk']}")

    user_question = st.text_area(
        "Ask the Copilot",
        value=f"Engine {selected} has RUL of {eng['mean_rul']:.0f} cycles "
              f"with 90% CI [{eng['lower_ci']:.0f}, {eng['upper_ci']:.0f}]. "
              f"What maintenance actions should be taken?",
        height=80
    )

    if st.button("🚀 Get Recommendation"):
        # Show retrieved docs
        with st.expander("📚 Retrieved Documentation Sources", expanded=False):
            context, sources = retriever.format_context(user_question)
            for s in sources:
                st.markdown(f"**[{s['source']}]** {s['title']}")
                st.caption(s['text'][:200] + "...")
                st.divider()

        # LLM response
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from groq import Groq
            import os

            api_key = os.getenv('GROQ_API_KEY')
            client  = Groq(api_key=api_key)

            from agent.copilot import SYSTEM_PROMPT
            prompt = f"""{SYSTEM_PROMPT}

ENGINE PREDICTION DATA:
- Engine ID: {selected}
- Mean RUL: {eng['mean_rul']:.1f} cycles
- 90% CI: [{eng['lower_ci']:.1f}, {eng['upper_ci']:.1f}] cycles
- Uncertainty (std): {eng['std']:.2f} cycles
- Risk Level: {eng['icon']} {eng['risk']}

{context}

USER QUESTION: {user_question}
"""
            with st.spinner("Consulting maintenance documentation..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )

            st.markdown("### 💡 Copilot Recommendation")
            st.markdown(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════════
# PAGE 4: Drift Monitor
# ════════════════════════════════════════════════════════════════
elif page == "Drift Monitor":
    st.markdown("# 📊 Model Drift Monitor")
    st.markdown("*Sensor distribution comparison — Training vs Production*")
    st.divider()

    if st.button("▶ Run Drift Detection Now"):
        with st.spinner("Running Evidently drift analysis..."):
            from monitoring.drift import load_data, run_drift_report
            reference, current = load_data()
            summary, _         = run_drift_report(
                reference, current, save=True
            )

        status_color = "#FF4444" if summary['drift_detected'] else "#00E676"
        status_text  = "🔴 DRIFT DETECTED" if summary['drift_detected'] \
                       else "✅ NO DRIFT"

        st.markdown(
            f'<div style="background:{status_color}22;border:1px solid '
            f'{status_color};padding:16px;border-radius:10px;text-align:center">'
            f'<h2 style="color:{status_color};margin:0">{status_text}</h2>'
            f'<p style="color:#aaa;margin:4px 0">'
            f'{summary["n_drifted"]} / {summary["n_features"]} '
            f'features drifted '
            f'({summary["drift_share"]*100:.0f}%)</p>'
            f'</div>', unsafe_allow_html=True
        )

        st.markdown("")
        if summary['feature_drifts']:
            drift_df = pd.DataFrame([
                {'Feature': k,
                 'Drift Score': round(v['drift_score'], 4),
                 'Drifted': '⚠ YES' if v['drift_detected'] else 'no',
                 'Test': v['stattest']}
                for k, v in sorted(
                    summary['feature_drifts'].items(),
                    key=lambda x: x[1]['drift_score'],
                    reverse=True
                )
            ])
            st.dataframe(drift_df, use_container_width=True,
                         hide_index=True)
    else:
        st.info("Click 'Run Drift Detection' to analyze sensor distributions.")
        st.markdown("**What this monitors:**")
        st.markdown("- Compares training sensor distributions vs incoming data")
        st.markdown("- Uses KS test per sensor channel")
        st.markdown("- Flags when >30% of features show significant drift")
        st.markdown("- Saves HTML report to `logs/drift_reports/`")


# ════════════════════════════════════════════════════════════════
# PAGE 5: Model Performance
# ════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("# 📈 Model Performance")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test RMSE",     "13.56 cycles", help="Lower is better")
    col2.metric("NASA Score",    "378.38",        help="Lower is better")
    col3.metric("ECE Score",     "0.2020",        help="Lower = better calibrated")
    col4.metric("90% CI Coverage", "54%",         delta="-36%",
                help="Ideal = 90%")

    st.divider()

    # Load calibration plot if exists
    cal_path = 'logs/calibration_report.png'
    if os.path.exists(cal_path):
        st.markdown("### Calibration Report")
        st.image(cal_path, use_column_width=True)
    else:
        st.info("Run `python evaluate.py` to generate calibration report.")

    st.divider()
    st.markdown("### What these metrics mean")
    st.markdown("""
    | Metric | Your Value | What it means |
    |--------|-----------|---------------|
    | RMSE | 13.56 cycles | Average prediction error. Competitive range for C-MAPSS FD001 is 12–18 |
    | NASA Score | 378 | Asymmetric score penalizing late predictions more. Lower = safer |
    | ECE | 0.2020 | Calibration error. 0 = perfect. Model is overconfident — known MC Dropout limitation |
    | CI Coverage | 54% | Only 54% of true RUL values fall inside the stated 90% CI. Confirms overconfidence |
    """)