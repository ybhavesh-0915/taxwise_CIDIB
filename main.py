
from fastapi import FastAPI, HTTPException
import requests
from typing import Dict, Any
import uvicorn
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import numpy as np
import os

# Railway Configuration
DATA_PROCESSOR_BASE_URL = os.getenv("DATA_PROCESSOR_URL", "http://localhost:8000")
PORT = int(os.getenv("PORT", 8001))

# Initialize FastAPI app
app = FastAPI(title="CIBIL Analysis Service", version="1.0.0")

# CIBIL calculation constants
CIBIL_FACTORS = {
    'payment_history': 0.35,
    'credit_utilization': 0.30,
    'credit_mix': 0.10,
    'debt_to_income': 0.15,
    'credit_inquiries': 0.10
}

SCORE_RANGES = {
    'excellent': {'min': 750, 'status': 'Excellent'},
    'good': {'min': 700, 'status': 'Good'},
    'fair': {'min': 650, 'status': 'Fair'},
    'poor': {'min': 300, 'status': 'Poor'}
}

def generate_score_chart(final_score: int, components: Dict) -> str:
    """Generate CIBIL score visualization"""
    try:
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'CIBIL Score: {final_score}', fontsize=16, fontweight='bold')
        
        # 1. Score Gauge
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_aspect('equal')
        
        colors = ['#ff4444', '#ff8800', '#ffdd00', '#44ff44']
        for i in range(len(colors)):
            start_angle = 180 - (i * 45)
            end_angle = 180 - ((i + 1) * 45)
            wedge = patches.Wedge((5, 3), 3, start_angle, end_angle, 
                                 facecolor=colors[i], alpha=0.7, width=0.8)
            ax1.add_patch(wedge)
        
        # Score needle
        score_angle = 180 - ((final_score - 300) / 600) * 180
        needle_x = 5 + 2.5 * np.cos(np.radians(score_angle))
        needle_y = 3 + 2.5 * np.sin(np.radians(score_angle))
        ax1.arrow(5, 3, needle_x-5, needle_y-3, head_width=0.2, head_length=0.2, 
                  fc='black', ec='black', linewidth=3)
        
        ax1.text(5, 1, str(final_score), ha='center', va='center', fontsize=24, fontweight='bold')
        ax1.set_title('Credit Score')
        ax1.axis('off')
        
        # 2. Component Breakdown
        comp_names = ['Payment\nHistory', 'Credit\nUtilization', 'Credit\nMix', 'Debt-to\nIncome', 'Credit\nInquiries']
        comp_scores = [components.get(name.replace('\n', ' '), 80) for name in comp_names]
        
        bars = ax2.barh(comp_names, comp_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
        ax2.set_xlim(0, 100)
        ax2.set_title('Score Components')
        
        for bar, score in zip(bars, comp_scores):
            ax2.text(score + 1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.0f}', va='center')
        
        # 3. Utilization Pie
        utilization = components.get('utilization_percentage', 30)
        ax3.pie([utilization, 100-utilization], 
                labels=[f'Used: {utilization}%', f'Available: {100-utilization}%'],
                colors=['#ff6b6b', '#e8e8e8'], startangle=90)
        ax3.set_title('Credit Utilization')
        
        # 4. Improvement Target
        targets = [final_score]
        labels = ['Current']
        if final_score < 750:
            targets.append(750)
            labels.append('Target')
        
        ax4.bar(labels, targets, color=['#ff6b6b', '#44ff44'][:len(targets)])
        ax4.set_ylim(300, 900)
        ax4.set_title('Improvement Target')
        
        for i, score in enumerate(targets):
            ax4.text(i, score + 10, str(score), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception:
        return ""

def analyze_cibil(data: Dict) -> Dict[str, Any]:
    """Core CIBIL analysis function"""
    credit_behavior = data.get('credit_behavior', {})
    transactions = data.get('relevant_transactions', [])
    session_id = data.get('session_id', 'unknown')
    
    # Categorize transactions
    home_loans = [t for t in transactions if "Home Loan" in t.get('description', '')]
    car_loans = [t for t in transactions if "Car Loan" in t.get('description', '')]
    life_insurance = [t for t in transactions if "Life Insurance" in t.get('description', '')]
    health_insurance = [t for t in transactions if "Health Insurance" in t.get('description', '')]
    cc_payments = [t for t in transactions if "Credit Card" in t.get('description', '')]
    
    # Calculate metrics
    total_cc_payments = sum(max(0, t['amount']) for t in cc_payments)
    estimated_cc_bills = max(total_cc_payments * 1.03, 50000)
    estimated_credit_limit = estimated_cc_bills * 3
    monthly_salary = 60000
    monthly_emi = credit_behavior.get('monthly_emi_burden', 0) / 12
    
    # 1. Payment History (35%)
    payment_consistency = 0.9
    payment_ratio = min(1.0, total_cc_payments / max(estimated_cc_bills, 1))
    payment_history_score = min(100, (payment_ratio * 80) + (payment_consistency * 20))
    
    # 2. Credit Utilization (30%)
    utilization_ratio = estimated_cc_bills / estimated_credit_limit
    if utilization_ratio <= 0.30:
        utilization_score = 90
    elif utilization_ratio <= 0.50:
        utilization_score = 90 - ((utilization_ratio - 0.30) / 0.20) * 20
    elif utilization_ratio <= 0.70:
        utilization_score = 70 - ((utilization_ratio - 0.50) / 0.20) * 20
    else:
        utilization_score = max(30, 50 - ((utilization_ratio - 0.70) / 0.30) * 20)
    
    # 3. Credit Mix (10%)
    credit_types = set()
    if home_loans: credit_types.add("Home")
    if car_loans: credit_types.add("Car")
    if life_insurance: credit_types.add("Life")
    if health_insurance: credit_types.add("Health")
    if cc_payments: credit_types.add("CC")
    
    credit_mix_score = min(100, len(credit_types) * 17)
    
    # 4. Debt-to-Income (15%)
    dti_ratio = monthly_emi / monthly_salary
    if dti_ratio <= 0.30:
        dti_score = 90
    elif dti_ratio <= 0.50:
        dti_score = 70
    else:
        dti_score = 50
    
    # 5. Credit Inquiries (10%)
    inquiry_score = 80
    
    # Calculate final score
    raw_score = (
        payment_history_score * CIBIL_FACTORS['payment_history'] +
        utilization_score * CIBIL_FACTORS['credit_utilization'] +
        credit_mix_score * CIBIL_FACTORS['credit_mix'] +
        dti_score * CIBIL_FACTORS['debt_to_income'] +
        inquiry_score * CIBIL_FACTORS['credit_inquiries']
    )
    
    final_score = int(300 + (raw_score / 100.0) * 600)
    
    # Determine status
    status = next((info['status'] for info in SCORE_RANGES.values() 
                  if info['min'] <= final_score), 'Unknown')
    
    # Generate advice
    advice = []
    if utilization_ratio > 0.30:
        advice.append("Reduce credit utilization below 30%")
    if monthly_emi > 30000:
        advice.append("Consider refinancing loans")
    if final_score < 750:
        advice.append("Maintain consistent payment history")
    
    # Create components for chart
    components = {
        'Payment History': payment_history_score,
        'Credit Utilization': utilization_score,
        'Credit Mix': credit_mix_score,
        'Debt-to Income': dti_score,
        'Credit Inquiries': inquiry_score,
        'utilization_percentage': int(utilization_ratio * 100)
    }
    
    chart = generate_score_chart(final_score, components)
    
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "cibil_score": final_score,
        "status": status,
        "components": {
            "payment_history": round(payment_history_score, 1),
            "credit_utilization": round(utilization_score, 1),
            "credit_mix": round(credit_mix_score, 1),
            "debt_to_income": round(dti_score, 1),
            "credit_inquiries": round(inquiry_score, 1)
        },
        "metrics": {
            "utilization_pct": f"{int(utilization_ratio * 100)}%",
            "dti_pct": f"{int(dti_ratio * 100)}%",
            "monthly_emi": f"₹{monthly_emi:,.0f}",
            "credit_types": len(credit_types)
        },
        "advice": advice,
        "chart_base64": chart
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "CIBIL Analysis"}

@app.get("/analyze-cibil/{session_id}")
async def analyze_cibil_score(session_id: str):
    """Single endpoint for CIBIL analysis"""
    try:
        url = f"{DATA_PROCESSOR_BASE_URL}/cibil-data/{session_id}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data")
        data = response.json()
        result = analyze_cibil(data)
        return result
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to data service")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)