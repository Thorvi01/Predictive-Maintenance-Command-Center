# agent/copilot.py
# Maintenance Copilot — connects RUL prediction + RAG retrieval + Groq LLM
# Flow: user query → get RUL prediction → retrieve relevant docs → LLM answer

import os
import sys
import torch
import numpy as np
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import RULPredictor, mc_predict
from rag.retriever import MaintenanceRetriever
from dataset import get_dataloaders

load_dotenv()

# Load temperature scalar for calibrated uncertainty
def load_temperature(path='models/temperature.pt'):
    if not os.path.exists(path):
        return 1.0
    data = torch.load(path, map_location='cpu', weights_only=False)
    return float(data['temperature'])

TEMPERATURE = load_temperature()


# ── 1. Load model ────────────────────────────────────────────────
def load_model(model_path='models/rul_predictor.pt', device='cpu'):
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False
    )
    config = checkpoint['config']
    model  = RULPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config


# ── 2. Risk tier classifier ──────────────────────────────────────
def classify_risk(mean_rul, lower_ci):
    effective_rul = min(mean_rul, lower_ci)
    if effective_rul < 10:
        return "CRITICAL", "🔴"
    elif effective_rul < 20:
        return "HIGH", "🟠"
    elif effective_rul < 50:
        return "ELEVATED", "🟡"
    elif effective_rul < 90:
        return "MODERATE", "🟢"
    else:
        return "HEALTHY", "✅"


# ── 3. System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert aircraft engine maintenance advisor
with deep knowledge of turbofan engine health monitoring and predictive
maintenance systems.

You have access to:
1. RUL (Remaining Useful Life) predictions from a calibrated LSTM model
   with Monte Carlo Dropout uncertainty quantification
2. Relevant maintenance documentation and NASA research

Your role is to provide clear, actionable maintenance recommendations.

Guidelines:
- Always reference the confidence interval, not just the mean RUL
- Use the LOWER bound of the CI for safety-critical scheduling decisions
- Cite the documentation sources you used
- Be specific about maintenance actions and urgency
- Flag high uncertainty (wide CI) as requiring human expert review
- Keep responses concise but complete — maintenance engineers are busy

Format your response as:
1. SITUATION SUMMARY (2-3 sentences)
2. RISK ASSESSMENT (one line)
3. RECOMMENDED ACTIONS (numbered list)
4. UNCERTAINTY NOTE (if CI is wide or coverage is low)
"""


# ── 4. Main Copilot class ────────────────────────────────────────
class MaintenanceCopilot:

    def __init__(self,
                 model_path='models/rul_predictor.pt',
                 n_mc_samples=100):

        print("Initializing Maintenance Copilot...")

        self.device = 'cpu'
        self.model, self.config = load_model(model_path, self.device)
        print("  ✓ RUL prediction model loaded")

        self.retriever = MaintenanceRetriever()
        print("  ✓ RAG retriever ready")

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)
        print("  ✓ Groq LLM connected (llama-3.3-70b)")

        self.n_mc_samples = n_mc_samples
        self.temperature  = TEMPERATURE
        print(f"  ✓ Temperature scaling: T={self.temperature:.4f}")
        print("\nCopilot ready. Ask me about any engine.\n")

    def predict_engine(self, engine_tensor):
        if engine_tensor.dim() == 2:
            engine_tensor = engine_tensor.unsqueeze(0)

        mean, std, lower, upper, _ = mc_predict(
            self.model,
            engine_tensor,
            n_samples=self.n_mc_samples,
            device=self.device
        )

        # Apply temperature scaling for calibrated uncertainty
        std_cal   = std * self.temperature
        lower_cal = mean - 1.645 * std_cal
        upper_cal = mean + 1.645 * std_cal

        return {
            'mean_rul': float(np.clip(mean[0],      0, None)),
            'std':      float(std_cal[0]),
            'lower_ci': float(np.clip(lower_cal[0], 0, None)),
            'upper_ci': float(upper_cal[0]),
            'ci_width': float(upper_cal[0] - lower_cal[0])
        }

    def get_recommendation(self, engine_id, prediction, user_question=None):
        mean_rul = prediction['mean_rul']
        lower_ci = prediction['lower_ci']
        upper_ci = prediction['upper_ci']
        std      = prediction['std']
        ci_width = prediction['ci_width']

        risk_level, risk_icon = classify_risk(mean_rul, lower_ci)

        if mean_rul < 20:
            query = "urgent maintenance action RUL less than 20 cycles critical"
        elif mean_rul < 50:
            query = f"maintenance planning RUL {int(mean_rul)} cycles elevated risk"
        else:
            query = f"monitoring recommendation RUL {int(mean_rul)} cycles healthy engine"

        if user_question:
            query = user_question

        context, sources = self.retriever.format_context(query, top_k=3)

        user_prompt = f"""ENGINE PREDICTION DATA:
- Engine ID: {engine_id}
- Mean RUL: {mean_rul:.1f} cycles
- 90% Confidence Interval: [{lower_ci:.1f}, {upper_ci:.1f}] cycles
- Uncertainty (std): {std:.2f} cycles
- CI Width: {ci_width:.1f} cycles
- Risk Level: {risk_icon} {risk_level}

{context}

USER QUESTION: {user_question or f'What maintenance action should be taken for Engine {engine_id}?'}

Please provide your maintenance recommendation based on the prediction data and documentation above.
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        return {
            'engine_id':  engine_id,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_icon':  risk_icon,
            'sources':    sources,
            'answer':     answer
        }

    def analyze_fleet(self, test_loader):
        print("Analyzing fleet...")
        all_predictions = []

        X_all = torch.cat([X for X, _ in test_loader])
        y_all = torch.cat([y for _, y in test_loader]).numpy()

        for i in range(len(X_all)):
            pred = self.predict_engine(X_all[i])
            risk, icon = classify_risk(pred['mean_rul'], pred['lower_ci'])
            all_predictions.append({
                'engine_id':  i + 1,
                'true_rul':   float(y_all[i]),
                'prediction': pred,
                'risk_level': risk,
                'risk_icon':  icon
            })

        all_predictions.sort(
            key=lambda x: min(
                x['prediction']['mean_rul'],
                x['prediction']['lower_ci']
            )
        )
        return all_predictions


# ── 5. Interactive demo ──────────────────────────────────────────
if __name__ == '__main__':
    copilot = MaintenanceCopilot()

    _, _, test_loader = get_dataloaders(
        batch_size=100,
        val_split=0.2,
        window_size=30
    )

    fleet = copilot.analyze_fleet(test_loader)

    print("\n" + "="*55)
    print("FLEET HEALTH SUMMARY (Most Critical First)")
    print("="*55)
    print(f"{'Engine':>8} {'True RUL':>10} {'Mean RUL':>10} "
          f"{'90% CI':>18} {'Risk':>12}")
    print("-" * 65)

    for eng in fleet[:10]:
        p = eng['prediction']
        print(
            f"{eng['engine_id']:>8} "
            f"{eng['true_rul']:>10.0f} "
            f"{p['mean_rul']:>10.1f} "
            f"[{p['lower_ci']:5.1f}, {p['upper_ci']:5.1f}]"
            f"{eng['risk_icon']:>6} {eng['risk_level']:>10}"
        )

    most_critical = fleet[0]
    engine_id     = most_critical['engine_id']
    prediction    = most_critical['prediction']

    print(f"\n{'='*55}")
    print(f"COPILOT RECOMMENDATION — Engine {engine_id}")
    print(f"{'='*55}")

    result = copilot.get_recommendation(
        engine_id=engine_id,
        prediction=prediction,
        user_question=(
            f"Engine {engine_id} has RUL of "
            f"{prediction['mean_rul']:.1f} cycles with "
            f"90% CI [{prediction['lower_ci']:.1f}, "
            f"{prediction['upper_ci']:.1f}]. "
            f"What immediate actions should maintenance crew take?"
        )
    )

    print(f"\nRisk: {result['risk_icon']} {result['risk_level']}")
    print(f"\nSources used:")
    for s in result['sources']:
        print(f"  - {s['source']}: {s['title'][:55]}")

    print(f"\n{result['answer']}")