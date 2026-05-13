# rag/documents.py
# Curated knowledge base for turbofan engine maintenance.
# In a production system these would be loaded from PDFs.
# For this portfolio project we use structured domain knowledge
# that accurately represents real maintenance documentation.

MAINTENANCE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Turbofan Engine RUL Interpretation Guide",
        "content": """
        Remaining Useful Life (RUL) represents the number of operational cycles
        remaining before an engine component requires maintenance or replacement.
        RUL predictions should be interpreted with their associated uncertainty bounds.

        Critical RUL thresholds for turbofan engines:
        - RUL > 90 cycles: Engine is healthy. Continue normal operations.
          Schedule routine inspection at next maintenance window.
        - RUL 50-90 cycles: Early warning zone. Increase monitoring frequency.
          Review sensor trends. Plan maintenance logistics.
        - RUL 20-50 cycles: Elevated risk zone. Prioritize this engine for
          maintenance scheduling. Avoid extending flight cycles unnecessarily.
        - RUL < 20 cycles: Critical zone. Immediate maintenance action required.
          Ground the engine if RUL < 10 cycles or uncertainty interval includes 0.

        Uncertainty interpretation:
        A wide confidence interval (CI > 30 cycles) indicates high model uncertainty,
        often due to unusual sensor patterns or operating conditions outside training data.
        When CI is wide, apply more conservative thresholds — treat the lower bound
        of the confidence interval as the effective RUL for scheduling decisions.
        """
    },
    {
        "id": "doc_002",
        "title": "NASA C-MAPSS Sensor Interpretation Manual",
        "content": """
        The C-MAPSS turbofan simulation models key sensors in a two-spool,
        mixed-flow turbofan engine. Critical sensors and their maintenance significance:

        T24 (s2) - Total temperature at fan inlet:
        Elevated T24 indicates potential issues with inlet guide vanes or fan blade erosion.
        Sudden changes correlate with foreign object damage events.

        T30 (s3) - Total temperature at HPC outlet:
        Rising T30 trend indicates HPC (High Pressure Compressor) degradation.
        T30 increase of >2% from baseline warrants borescope inspection.

        T50 (s4) - Total temperature at LPT outlet:
        T50 trends reflect LPT (Low Pressure Turbine) efficiency degradation.
        Primary indicator for turbine blade wear and thermal barrier coating erosion.

        P30 (s7) - Total pressure at HPC outlet:
        Declining P30 indicates compressor stall margin reduction.
        Critical for surge prediction. P30 drop >3% triggers immediate inspection.

        Nf (s11) - Physical fan speed:
        Fan speed deviations indicate bearing wear or rotor imbalance.
        Vibration correlation required for diagnosis.

        NRc (s12) - Corrected core speed:
        Core speed changes reflect turbine efficiency loss.
        Trending NRc against T30 identifies compressor vs turbine degradation source.

        BPR (s13) - Bypass ratio:
        Changing bypass ratio indicates fan performance degradation.
        Fan blade erosion and tip clearance increase reduce BPR.
        """
    },
    {
        "id": "doc_003",
        "title": "Predictive Maintenance Action Protocols",
        "content": """
        Standard maintenance action protocols based on degradation severity:

        LEVEL 1 — Monitoring (RUL > 90):
        Action: Continue normal operation
        Monitoring: Standard sensor logging at 1Hz
        Documentation: Log prediction in maintenance system
        Next review: Scheduled maintenance interval

        LEVEL 2 — Enhanced Monitoring (RUL 50-90):
        Action: Increase sensor sampling to 10Hz for critical sensors
        Inspection: Visual inspection of accessible components
        Documentation: Flag in maintenance tracking system
        Parts: Order replacement parts to ensure availability
        Next review: Every 10 flight cycles

        LEVEL 3 — Maintenance Planning (RUL 20-50):
        Action: Schedule maintenance within next 15 cycles
        Inspection: Borescope inspection of HPC and LPT sections
        Oil analysis: Conduct spectrometric oil analysis for metal particles
        Parts: Confirm parts availability and maintenance crew scheduling
        Next review: Every 5 flight cycles

        LEVEL 4 — Urgent Maintenance (RUL < 20):
        Action: Complete maintenance within 5 cycles maximum
        Inspection: Full engine teardown inspection
        Replacement: Replace all life-limited parts approaching limits
        Documentation: Submit airworthiness directive compliance report
        Next review: Post-maintenance test flight required

        LEVEL 5 — Ground Engine (RUL < 10 or lower CI bound < 5):
        Action: Remove from service immediately
        Justification: Safety risk unacceptable for continued operation
        Process: Follow airline operations manual section 5.3
        """
    },
    {
        "id": "doc_004",
        "title": "MC Dropout Uncertainty Quantification in Maintenance Decisions",
        "content": """
        Monte Carlo Dropout provides probabilistic RUL estimates that should
        directly influence maintenance decision-making.

        Understanding the confidence interval:
        The 90% confidence interval [lower, upper] means that in 90% of similar
        cases, the true RUL falls within this range. For safety-critical decisions,
        always use the LOWER bound of the CI for scheduling.

        Decision rules with uncertainty:

        Rule 1 — Conservative scheduling:
        If lower_CI < threshold: treat as if RUL = lower_CI
        Example: Mean RUL = 45, CI = [28, 62] → schedule as if RUL = 28

        Rule 2 — High uncertainty flag:
        If (upper_CI - lower_CI) > 40 cycles: flag for human expert review
        Wide intervals indicate the model has low confidence — sensor anomaly
        or operating condition outside training distribution likely.

        Rule 3 — Imminent failure detection:
        If lower_CI < 10: ground engine regardless of mean prediction
        Safety takes priority over operational efficiency.

        Rule 4 — Uncertainty trend monitoring:
        If std_dev increases significantly over consecutive predictions:
        This indicates model is detecting unusual degradation patterns.
        Escalate monitoring level regardless of mean RUL value.

        Calibration note:
        The MC Dropout model on C-MAPSS FD001 shows ECE of 0.20,
        indicating overconfident intervals. Apply a safety factor of 1.5x
        to the stated uncertainty when making critical decisions.
        """
    },
    {
        "id": "doc_005",
        "title": "Engine Degradation Modes and Sensor Signatures",
        "content": """
        Common turbofan degradation modes and their sensor signatures in C-MAPSS:

        1. HPC Degradation (most common in FD001):
        Primary sensors: s3 (T30), s7 (P30), s12 (NRc)
        Signature: T30 increases, P30 decreases, NRc drifts
        Physical cause: Blade tip clearance increase, erosion, fouling
        Typical progression: Gradual over 50-100 cycles
        Maintenance action: HPC cleaning or blade replacement

        2. Fan Degradation:
        Primary sensors: s2 (T24), s11 (Nf), s13 (BPR)
        Signature: T24 rise, Nf instability, BPR reduction
        Physical cause: Fan blade erosion, inlet distortion
        Typical progression: Can be sudden after foreign object ingestion
        Maintenance action: Fan blade inspection and replacement

        3. LPT Efficiency Loss:
        Primary sensors: s4 (T50), s9 (Ps30), s14 (farB)
        Signature: T50 increases with T30 relatively stable
        Physical cause: Turbine blade wear, tip seal degradation
        Typical progression: Gradual, accelerates in later life
        Maintenance action: LPT module replacement

        4. Combustor Degradation:
        Primary sensors: s3 (T30), s4 (T50), s8 (Ps30)
        Signature: T50/T30 ratio changes, fuel flow increases
        Physical cause: Combustor liner cracking, fuel nozzle coking
        Typical progression: Can be rapid
        Maintenance action: Combustor inspection and liner replacement

        Degradation interaction:
        In practice, multiple degradation modes occur simultaneously.
        The LSTM model captures combined multi-sensor patterns,
        which is why it outperforms single-sensor threshold methods.
        """
    },
    {
        "id": "doc_006",
        "title": "Fleet Maintenance Optimization Principles",
        "content": """
        Fleet-level maintenance optimization using RUL predictions:

        Scheduling principles:
        1. Batch maintenance: Group engines with similar RUL for simultaneous
           shop visits to maximize maintenance crew efficiency
        2. AOG prevention: Always prioritize engines where lower CI < 15 cycles
           to prevent Aircraft on Ground (AOG) situations
        3. Parts pooling: Use RUL predictions to pre-position spare parts
           at maintenance bases before engines arrive

        Cost-benefit framework:
        Preventive maintenance cost: ~$50,000 per shop visit
        Unplanned maintenance cost: ~$500,000 per AOG event
        Therefore: False negatives (missed failures) cost 10x more than
        false positives (unnecessary early maintenance)
        This justifies using lower CI bound for all scheduling decisions.

        Maintenance interval optimization:
        Traditional time-based maintenance: Every 500 cycles regardless of health
        Condition-based maintenance using RUL: Extends healthy engines by 15-30%
        Expected annual savings per 10-engine fleet: $200,000-$400,000

        Data quality requirements:
        RUL predictions are only valid when:
        - All 14 sensor channels are operational (no missing data)
        - Operating conditions match training distribution
        - Model drift monitor shows no significant distribution shift
        If any condition is violated, fall back to conservative time-based schedule.
        """
    },
    {
        "id": "doc_007",
        "title": "Model Drift Detection and Retraining Protocols",
        "content": """
        Production ML models for maintenance prediction require continuous monitoring.

        Types of drift in RUL prediction systems:

        1. Data drift (covariate shift):
        Definition: Input sensor distributions change from training data
        Detection: Statistical tests (KS test, PSI) on sensor distributions
        Cause: Different engine variants, seasonal effects, sensor calibration drift
        Action: Retrain model on new data if drift score > 0.1

        2. Concept drift:
        Definition: Relationship between sensors and RUL changes
        Detection: Rolling window RMSE increase, CI coverage degradation
        Cause: New failure modes, maintenance policy changes, fleet aging
        Action: Collect labeled failure data and retrain

        3. Prediction drift:
        Definition: Model output distribution shifts over time
        Detection: Monitor mean and std of predictions on incoming data
        Cause: Usually indicates data drift upstream
        Action: Investigate sensor pipeline before retraining

        Retraining triggers:
        - Data drift score (PSI) > 0.2 on any critical sensor
        - Rolling 30-day RMSE increases > 15% above baseline
        - 90% CI coverage drops below 70% on recent validated cases
        - Manual trigger by maintenance engineer

        Retraining protocol:
        1. Collect new labeled data (minimum 20 new engine run-to-failure cycles)
        2. Combine with original training data (weighted 70% new, 30% old)
        3. Retrain with same architecture, tune dropout rate if calibration poor
        4. Validate: new model must beat current model on held-out test set
        5. Champion/challenger: deploy new model only if RMSE improves > 2%
        6. Log all metrics and data lineage in MLflow for audit trail
        """
    },
]