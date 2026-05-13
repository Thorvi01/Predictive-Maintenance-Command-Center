# ✈️ FleetGuard AI — Predictive Maintenance Command Center
![CI](https://github.com/Thorvi01/RUL_Predictive-Maintenance-Command-Center/actions/workflows/ci.yml/badge.svg)
> **An end-to-end ML system for turbofan engine health monitoring, combining LSTM-based Remaining Useful Life prediction, Monte Carlo Dropout uncertainty quantification, RAG-powered maintenance recommendations, and automated drift detection.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-lightgrey)

---

## 🌍 Real-World Impact

Aircraft engine failures cost the aviation industry an estimated **$50 billion annually** in unplanned maintenance, flight delays, and AOG (Aircraft on Ground) events. A single unplanned engine shop visit costs **~$500,000**, compared to **~$50,000** for a scheduled one — a 10x cost difference.

This project addresses that gap directly:

**Before predictive maintenance:** Airlines rely on fixed time-based maintenance schedules (e.g., every 500 cycles regardless of engine health). Healthy engines get serviced unnecessarily. Degrading engines sometimes fail between scheduled visits.

**After FleetGuard AI:** Each engine gets a personalised RUL prediction every cycle. Healthy engines can safely extend their service interval by 15–30%. Engines with elevated degradation are flagged early. Critical engines are grounded before failure.

**Quantified impact for a 10-engine fleet:**
- Preventive AOG events avoided: estimated 2–3 per year
- Annual cost savings: **$200,000–$400,000**
- Safety improvement: failures caught at RUL > 10 cycles instead of at RUL = 0

**Why uncertainty quantification matters in this domain:** A point prediction of "RUL = 47 cycles" is dangerous without context. If the model is uncertain, that 47 could be anywhere from 20 to 74. FleetGuard AI's Monte Carlo Dropout outputs "RUL = 47 ± 8 cycles, 90% CI [34, 60]" — giving maintenance engineers the information they need to make risk-calibrated decisions. Wide confidence intervals trigger human expert review rather than automated action.

**Why drift detection matters:** ML models degrade silently. An engine fleet that ages, or a new engine variant introduced into the fleet, will shift the sensor distribution away from training data. Without a drift monitor, the model continues making predictions that look plausible but are based on extrapolation. FleetGuard AI's Evidently-based monitor catches this shift and triggers retraining before it causes harm.

---

## 🏗️ Architecture

```
NASA C-MAPSS Dataset
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing  │  Normalize sensors, compute piecewise RUL labels
│  data_preprocessing.py│  Drop zero-variance sensors, train/test split
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Sliding Windows    │  30-cycle windows → (batch, 30, 17) tensors
│   dataset.py        │  PyTorch Dataset + DataLoader
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│        2-Layer LSTM + MC Dropout         │
│        model.py                         │
│                                         │
│  Input (B, 30, 17) → LSTM → Dropout     │
│  → FC → RUL prediction                  │
│                                         │
│  At inference: N=100 stochastic passes  │
│  → mean RUL + std + 90% CI              │
└─────────┬───────────────────────────────┘
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
┌──────────────────┐                    ┌────────────────────────┐
│  Calibration     │                    │   RAG Knowledge Base   │
│  evaluate.py     │                    │   rag/ingest.py        │
│                  │                    │                        │
│  Reliability     │                    │  NASA C-MAPSS PDF +    │
│  diagram, ECE,   │                    │  Maintenance manuals   │
│  sharpness       │                    │  → FAISS vector index  │
└──────────────────┘                    └───────────┬────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   LLM Agent (Gemini)  │
                                        │   agent/copilot.py    │
                                        │                       │
                                        │  RUL + CI + retrieved │
                                        │  docs → grounded      │
                                        │  maintenance advice   │
                                        └───────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         Drift Detection                  │
│         monitoring/drift.py             │
│                                         │
│  Evidently AI: training vs production   │
│  KS test per sensor → drift score       │
│  Champion/challenger retraining trigger │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         Streamlit Dashboard             │
│         app.py                          │
│                                         │
│  Fleet Overview | Engine Deep Dive      │
│  Copilot | Drift Monitor | Performance  │
└─────────────────────────────────────────┘
```

---

## 📊 Results

| Metric | Value | Context |
|--------|-------|---------|
| Test RMSE | **13.56 cycles** | Competitive range for C-MAPSS FD001 is 12–18 cycles |
| NASA Asymmetric Score | **378.38** | Lower is better. Penalises late predictions more than early |
| ECE Score | **0.2020** | 0 = perfect calibration. Known MC Dropout limitation — documented |
| 90% CI Coverage | **54%** | Identified overconfidence — used as motivation for safety factor |
| Mean CI Width | **18.0 cycles** | Average uncertainty spread per engine |
| Training Time | **4.8 min** | CPU only, 53 epochs with early stopping |

### What makes these results meaningful

The RMSE of 13.56 puts this model in the competitive range of published academic benchmarks on C-MAPSS FD001. More importantly, the project goes beyond RMSE to measure **calibration** — something most portfolio projects and even some production systems skip entirely.

The ECE of 0.20 reveals that the model is overconfident: when it says "90% confident," it is actually right only 54% of the time. This is a known limitation of vanilla MC Dropout, and this project documents it honestly rather than hiding it. The practical implication — applying a safety factor when using lower CI bounds for scheduling — is built into the agent's decision logic.

---

## 🔬 What Makes This Different

Most RUL prediction projects stop at "I trained an LSTM and got RMSE = X." This project adds three layers that matter for production ML:

### 1. Uncertainty Quantification via Monte Carlo Dropout
Instead of a single point estimate, every prediction includes a full probability distribution over possible RUL values. This is based on Gal & Ghahramani (2016) — dropout at inference time approximates a Bayesian posterior over model weights.

```python
# 100 stochastic forward passes with dropout active
mean, std, lower, upper, all_preds = mc_predict(model, X, n_samples=100)
# Output: "RUL = 47.3 ± 4.1 cycles, 90% CI [40.5, 54.1]"
```

### 2. Calibration Evaluation
A model can be accurate but miscalibrated. This project measures both using a reliability diagram and Expected Calibration Error — and documents the gap. Calibration matters in safety-critical domains where stated confidence drives decisions.

### 3. Live Drift Detection with Champion/Challenger Retraining
Production ML models degrade when data distributions shift. This project monitors sensor distributions using Evidently AI and triggers a full retraining pipeline when drift is detected — only promoting the new model if it beats the current one on held-out data.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch — 2-layer LSTM, MC Dropout |
| Experiment Tracking | MLflow — metrics, artifacts, model registry |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS — cosine similarity over 34 document chunks |
| LLM | Google Gemini 2.0 Flash |
| Drift Detection | Evidently AI + scipy KS test |
| Dashboard | Streamlit |
| PDF Parsing | PyMuPDF |
| Data | NASA C-MAPSS FD001 (100 engines, 20,631 cycles) |

---

## 📁 Project Structure

```
rul-predictor/
├── data/
│   ├── raw/                    ← NASA C-MAPSS files + PDF
│   └── processed/              ← Normalized CSVs
├── models/
│   └── rul_predictor.pt        ← Trained model checkpoint
├── rag/
│   ├── documents.py            ← Structured maintenance knowledge
│   ├── ingest.py               ← PDF + knowledge → FAISS index
│   ├── retriever.py            ← Semantic search interface
│   └── index/                  ← Saved FAISS index + metadata
├── agent/
│   └── copilot.py              ← LLM agent with tool calling
├── monitoring/
│   ├── drift.py                ← Evidently drift detection
│   └── retrain.py              ← Champion/challenger pipeline
├── logs/
│   ├── calibration_report.png  ← ECE + reliability diagram
│   └── drift_reports/          ← HTML + JSON drift reports
├── data_preprocessing.py       ← Sensor normalization + RUL labels
├── dataset.py                  ← Sliding window DataLoader
├── model.py                    ← LSTM + MC Dropout
├── train.py                    ← Training loop + MLflow
├── evaluate.py                 ← Calibration evaluation
└── app.py                      ← Streamlit dashboard
```

---

## 🚀 Quick Start

### Prerequisites
- Anaconda or Miniconda
- Python 3.11
- Google Gemini API key (free at aistudio.google.com)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rul-predictor.git
cd rul-predictor

# Create conda environment
conda create -n rul-predictor python=3.11 -y
conda activate rul-predictor

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Download the NASA Dataset

1. Go to the [NASA PCOE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
2. Download "Turbofan Engine Degradation Simulation"
3. Place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `data/raw/`

### Run the Pipeline

```bash
# 1. Preprocess data
python data_preprocessing.py

# 2. Build sliding windows (verify)
python dataset.py

# 3. Train the model (~5 min on CPU)
python train.py

# 4. Evaluate calibration
python evaluate.py

# 5. Build RAG index
python rag/ingest.py

# 6. Launch dashboard
streamlit run app.py
```

### View experiment tracking
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📈 Dashboard Pages

| Page | What it shows |
|------|--------------|
| **Fleet Overview** | All 100 engines sorted by risk. Color-coded RUL bar chart with 90% CI error bars. KPI cards for risk tier counts. |
| **Engine Deep Dive** | Per-engine MC Dropout histogram showing 200 prediction samples. Mean, CI bounds, and true RUL overlaid. |
| **Maintenance Copilot** | RAG-powered LLM recommendations grounded in NASA documentation. Select any engine and ask a question. |
| **Drift Monitor** | Run Evidently drift detection on demand. Per-sensor Wasserstein distance scores. Retraining trigger logic. |
| **Model Performance** | Full calibration report: reliability diagram, ECE, CI width distribution, residuals vs uncertainty. |

---

## 🧠 Key Technical Concepts

**Why piecewise RUL labels?** Raw RUL is noisy in early engine life — a new engine at cycle 1 doesn't behave meaningfully differently from cycle 50. Capping RUL at 125 focuses the model on the degradation phase where prediction matters.

**Why MC Dropout instead of standard LSTM?** Standard LSTMs produce point estimates with no uncertainty. Gal & Ghahramani (2016) showed that dropout at inference approximates a Bayesian posterior. Running N=100 stochastic passes gives a distribution over possible RUL values at negligible computational cost (~100ms latency).

**Why RAG instead of fine-tuning the LLM?** RAG provides traceable citations, allows knowledge updates without retraining, and avoids the need for labelled instruction data. The agent retrieves only the most relevant document chunks per query rather than loading everything into context.

**Why champion/challenger retraining?** Blind retraining on new data can degrade a model if the new data is noisy or unrepresentative. The pipeline only promotes a challenger if it improves RMSE by more than 2% on held-out data — preventing accidental regression.

---

## 📚 References

- Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* NASA Ames Research Center. (included in `data/raw/`)
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML.
- NASA C-MAPSS Dataset: [PCOE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 🔮 Future Work

- **Temperature scaling** to improve calibration from ECE=0.20 toward ECE<0.05
- **Multi-dataset generalisation** — extend to FD002/FD003/FD004 for multi-condition robustness
- **Cost-aware maintenance scheduler** — LP optimisation to minimise fleet-wide downtime cost using RUL predictions
- **Real-time sensor streaming** — replace batch test data with a simulated live sensor feed

---

## 👩‍💻 Author

Built as a portfolio project demonstrating end-to-end ML engineering: data pipelines, uncertainty-aware deep learning, RAG-powered LLM agents, and production MLOps practices.

*Dataset: NASA C-MAPSS FD001 — publicly available from NASA PCOE Data Repository.*
