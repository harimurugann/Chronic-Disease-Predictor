# agents/swarm_logic.py
import numpy as np

class DiagnosticSwarm:
    def __init__(self):
        # Initialize the swarm. 
        # Future update: Load actual joblib/pickle models here.
        # e.g., self.cardio_model = joblib.load('artifacts/cardio_rf.pkl')
        self.is_ready = True

    def cardio_agent(self, data):
        # Simulated Expert System for Cardiology
        # Replaces with actual model.predict_proba() later
        risk = 0.10
        if data['BloodPressure'] > 130: risk += 0.35
        if data['Age'] > 50: risk += 0.20
        if data['Smoking'] == 'Yes': risk += 0.15
        return min(risk, 0.95)

    def diabetic_agent(self, data):
        # Simulated Expert System for Endocrinology (Diabetes)
        risk = 0.05
        if data['Glucose'] > 125: risk += 0.45
        if data['BMI'] > 28.0: risk += 0.25
        if data['PhysicalActivity'] < 3.0: risk += 0.15
        return min(risk, 0.92)

    def chronic_agent(self, data):
        # Simulated Global Chronic Disease Agent
        risk = 0.10
        if data['StressLevel'] > 7: risk += 0.20
        if data['Sleep'] < 5.0: risk += 0.15
        if data['Diet'] == 'Poor': risk += 0.25
        return min(risk, 0.88)

    def get_swarm_consensus(self, patient_data):
        """
        Executes all AI agents in parallel and aggregates their findings 
        to form a final diagnostic consensus.
        """
        # 1. Fetch individual agent predictions
        cardio_risk = self.cardio_agent(patient_data)
        diabetic_risk = self.diabetic_agent(patient_data)
        chronic_risk = self.chronic_agent(patient_data)

        # 2. Swarm Logic: Weighted Average for Overall Health Score
        # Can be upgraded to Meta-Learner (Stacking) in the future
        agents = [cardio_risk, diabetic_risk, chronic_risk]
        overall_risk = np.mean(agents)
        critical_alert = any(risk > 0.75 for risk in agents)

        return {
            "Cardio_Score": cardio_risk,
            "Diabetic_Score": diabetic_risk,
            "Chronic_Score": chronic_risk,
            "Overall_Consensus": overall_risk,
            "Critical_Alert": critical_alert
        }