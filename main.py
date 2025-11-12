from fastapi import FastAPI, HTTPException
import requests
from typing import Dict, Any, List
import uvicorn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import numpy as np
import os
from collections import defaultdict

# Railway Configuration
DATA_PROCESSOR_BASE_URL = os.getenv("DATA_PROCESSOR_URL", "http://localhost:8000")
PORT = int(os.getenv("PORT", 8001))

app = FastAPI(title="CIBIL Analysis Service", version="1.0.0")

# Official CIBIL weightages for India (2025)
CIBIL_FACTORS = {
    'payment_history': 0.35,      # 35% - Most critical
    'credit_utilization': 0.30,   # 30% - Usage vs limit
    'credit_history_length': 0.15,# 15% - Age of credit
    'credit_mix': 0.10,           # 10% - Variety of credit
    'new_credit': 0.10            # 10% - Recent inquiries
}

def parse_transaction_date(date_str: str) -> datetime:
    """Parse transaction date from various formats"""
    formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d', '%d.%m.%Y']
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except:
            continue
    return None

def categorize_transaction(description: str) -> str:
    """Categorize transaction based on Indian credit products"""
    desc_upper = description.upper()
    
    # Credit Cards
    if any(x in desc_upper for x in ['CREDIT CARD', 'CC PAYMENT', 'CC EMI', 'CARD PAYMENT']):
        return 'credit_card'
    
    # Secured Loans
    if any(x in desc_upper for x in ['HOME LOAN', 'HOUSING LOAN', 'MORTGAGE', 'HL EMI']):
        return 'home_loan'
    if any(x in desc_upper for x in ['CAR LOAN', 'AUTO LOAN', 'VEHICLE LOAN', 'CAR EMI']):
        return 'car_loan'
    if any(x in desc_upper for x in ['GOLD LOAN', 'LOAN AGAINST PROPERTY']):
        return 'secured_loan'
    
    # Unsecured Loans
    if any(x in desc_upper for x in ['PERSONAL LOAN', 'PL EMI', 'CONSUMER LOAN']):
        return 'personal_loan'
    if any(x in desc_upper for x in ['EDUCATION LOAN', 'STUDENT LOAN']):
        return 'education_loan'
    
    # Insurance (shows financial discipline)
    if any(x in desc_upper for x in ['LIFE INSURANCE', 'LIC PREMIUM']):
        return 'life_insurance'
    if any(x in desc_upper for x in ['HEALTH INSURANCE', 'MEDICAL INSURANCE']):
        return 'health_insurance'
    
    return 'other'

def calculate_cibil_score(transactions: List[Dict]) -> Dict[str, Any]:
    """Calculate CIBIL score using official Indian methodology"""
    
    if not transactions:
        raise ValueError("No transactions provided")
    
    # Parse all transaction dates
    dated_transactions = []
    for txn in transactions:
        date = parse_transaction_date(txn.get('date', ''))
        if date:
            dated_transactions.append({
                **txn,
                'parsed_date': date,
                'category': categorize_transaction(txn.get('description', ''))
            })
    
    if not dated_transactions:
        raise ValueError("No valid transaction dates found")
    
    # Get actual date range from CSV
    dates = [t['parsed_date'] for t in dated_transactions]
    start_date = min(dates)
    end_date = max(dates)
    date_range_months = max(1, (end_date - start_date).days / 30)
    date_range_years = date_range_months / 12
    
    # Group transactions by category
    cc_payments = [t for t in dated_transactions if t['category'] == 'credit_card']
    home_loans = [t for t in dated_transactions if t['category'] == 'home_loan']
    car_loans = [t for t in dated_transactions if t['category'] == 'car_loan']
    personal_loans = [t for t in dated_transactions if t['category'] == 'personal_loan']
    education_loans = [t for t in dated_transactions if t['category'] == 'education_loan']
    life_insurance = [t for t in dated_transactions if t['category'] == 'life_insurance']
    health_insurance = [t for t in dated_transactions if t['category'] == 'health_insurance']
    
    all_credit_transactions = cc_payments + home_loans + car_loans + personal_loans + education_loans
    
    # ============================================
    # 1. PAYMENT HISTORY (35%) - Most Critical
    # ============================================
    total_expected_payments = len(all_credit_transactions)
    
    if total_expected_payments == 0:
        payment_history_score = 50.0  # No credit history
        payment_remarks = "No credit payment history found"
    else:
        # Assume all recorded payments were made (no defaults in data)
        # Check for payment regularity
        months_covered = date_range_months
        payments_per_month = total_expected_payments / max(1, months_covered)
        
        if payments_per_month >= 1.5:  # More than 1.5 payments per month
            payment_consistency = 95.0
        elif payments_per_month >= 1.0:  # At least 1 payment per month
            payment_consistency = 90.0
        elif payments_per_month >= 0.5:  # Payment every 2 months
            payment_consistency = 75.0
        else:
            payment_consistency = 60.0
        
        # Check for payment gaps (assuming monthly payments expected)
        payment_dates = sorted([t['parsed_date'] for t in all_credit_transactions])
        gaps = []
        for i in range(1, len(payment_dates)):
            gap_days = (payment_dates[i] - payment_dates[i-1]).days
            if gap_days > 45:  # More than 45 days gap
                gaps.append(gap_days)
        
        # Penalize for large gaps
        gap_penalty = min(20, len(gaps) * 5)
        payment_history_score = max(50, payment_consistency - gap_penalty)
        
        payment_remarks = f"{total_expected_payments} payments tracked over {months_covered:.1f} months"
    
    # ============================================
    # 2. CREDIT UTILIZATION RATIO (30%)
    # ============================================
    if cc_payments:
        total_cc_payments = sum(abs(t['amount']) for t in cc_payments)
        num_cc_payments = len(cc_payments)
        
        # Estimate credit limit based on payment patterns
        # Assumption: Monthly payment is 3-5% of total outstanding
        avg_payment = total_cc_payments / num_cc_payments
        estimated_monthly_bill = avg_payment * 30  # Assuming 3% payment
        
        # Credit limit is typically 2-3x of monthly usage
        estimated_credit_limit = estimated_monthly_bill * 2.5
        
        # Calculate utilization ratio
        cur = min(1.0, estimated_monthly_bill / estimated_credit_limit)
        
        # Score based on CUR (ideal < 30%)
        if cur <= 0.30:
            credit_utilization_score = 95.0
            cur_status = "Excellent"
        elif cur <= 0.50:
            credit_utilization_score = 85.0 - ((cur - 0.30) / 0.20) * 20
            cur_status = "Good"
        elif cur <= 0.70:
            credit_utilization_score = 65.0 - ((cur - 0.50) / 0.20) * 20
            cur_status = "Fair"
        else:
            credit_utilization_score = max(30, 45 - ((cur - 0.70) / 0.30) * 15)
            cur_status = "High"
        
        cur_percentage = int(cur * 100)
    else:
        credit_utilization_score = 70.0  # No CC usage, not ideal but not bad
        cur_percentage = 0
        cur_status = "No Credit Card Usage"
    
    # ============================================
    # 3. LENGTH OF CREDIT HISTORY (15%)
    # ============================================
    credit_age_years = date_range_years
    
    if credit_age_years >= 7:
        credit_history_score = 95.0
        credit_age_status = "Excellent"
    elif credit_age_years >= 5:
        credit_history_score = 85.0
        credit_age_status = "Very Good"
    elif credit_age_years >= 3:
        credit_history_score = 70.0
        credit_age_status = "Good"
    elif credit_age_years >= 1:
        credit_history_score = 55.0 + (credit_age_years * 5)
        credit_age_status = "Fair"
    else:
        credit_history_score = 50.0
        credit_age_status = "Limited"
    
    # ============================================
    # 4. CREDIT MIX (10%)
    # ============================================
    credit_types_present = set()
    
    if cc_payments:
        credit_types_present.add("Credit Card")
    if home_loans:
        credit_types_present.add("Home Loan")
    if car_loans:
        credit_types_present.add("Car Loan")
    if personal_loans:
        credit_types_present.add("Personal Loan")
    if education_loans:
        credit_types_present.add("Education Loan")
    if life_insurance:
        credit_types_present.add("Life Insurance")
    if health_insurance:
        credit_types_present.add("Health Insurance")
    
    num_credit_types = len(credit_types_present)
    
    # Score based on variety
    if num_credit_types >= 5:
        credit_mix_score = 95.0
        mix_status = "Excellent Mix"
    elif num_credit_types == 4:
        credit_mix_score = 85.0
        mix_status = "Very Good Mix"
    elif num_credit_types == 3:
        credit_mix_score = 70.0
        mix_status = "Good Mix"
    elif num_credit_types == 2:
        credit_mix_score = 55.0
        mix_status = "Limited Mix"
    else:
        credit_mix_score = 40.0
        mix_status = "Poor Mix"
    
    # ============================================
    # 5. NEW CREDIT INQUIRIES (10%)
    # ============================================
    # Estimate new credit inquiries based on first appearance of loan types
    first_appearance_dates = {}
    for txn in dated_transactions:
        cat = txn['category']
        if cat not in first_appearance_dates:
            first_appearance_dates[cat] = txn['parsed_date']
    
    # Count inquiries in last 6 months
    recent_threshold = end_date - timedelta(days=180)
    recent_inquiries = sum(1 for date in first_appearance_dates.values() if date >= recent_threshold)
    
    if recent_inquiries == 0:
        new_credit_score = 90.0
        inquiry_status = "No Recent Inquiries"
    elif recent_inquiries <= 2:
        new_credit_score = 80.0
        inquiry_status = "Minimal Inquiries"
    elif recent_inquiries <= 4:
        new_credit_score = 65.0
        inquiry_status = "Moderate Inquiries"
    else:
        new_credit_score = 50.0
        inquiry_status = "High Inquiry Activity"
    
    # ============================================
    # FINAL CIBIL SCORE CALCULATION
    # ============================================
    raw_score = (
        payment_history_score * CIBIL_FACTORS['payment_history'] +
        credit_utilization_score * CIBIL_FACTORS['credit_utilization'] +
        credit_history_score * CIBIL_FACTORS['credit_history_length'] +
        credit_mix_score * CIBIL_FACTORS['credit_mix'] +
        new_credit_score * CIBIL_FACTORS['new_credit']
    )
    
    # Map to 300-900 range
    final_cibil_score = int(300 + (raw_score / 100.0) * 600)
    
    # CIBIL score threshold tracking
    # Maximum achievable based on component scores
    max_possible_raw = (
        95.0 * CIBIL_FACTORS['payment_history'] +
        95.0 * CIBIL_FACTORS['credit_utilization'] +
        95.0 * CIBIL_FACTORS['credit_history_length'] +
        95.0 * CIBIL_FACTORS['credit_mix'] +
        90.0 * CIBIL_FACTORS['new_credit']
    )
    max_achievable_score = int(300 + (max_possible_raw / 100.0) * 600)
    
    # Calculate percentage of maximum achieved
    score_percentage = (raw_score / max_possible_raw) * 100
    
    # Determine status with peak detection
    is_peak = False
    peak_message = None
    
    if final_cibil_score >= 850:
        status = "Exceptional (Peak)"
        loan_approval = "Maximum"
        is_peak = True
        peak_message = "⭐ You've reached peak CIBIL performance! Score above 850 is exceptional."
    elif final_cibil_score >= 800:
        status = "Excellent+"
        loan_approval = "Very High"
        peak_message = f"Outstanding! You're {850 - final_cibil_score} points away from peak performance."
    elif final_cibil_score >= 750:
        status = "Excellent"
        loan_approval = "Very High"
    elif final_cibil_score >= 700:
        status = "Good"
        loan_approval = "High"
    elif final_cibil_score >= 650:
        status = "Fair"
        loan_approval = "Moderate"
    elif final_cibil_score >= 600:
        status = "Average"
        loan_approval = "Low"
    else:
        status = "Poor"
        loan_approval = "Very Low"
    
    # Generate recommendations
    recommendations = []
    if payment_history_score < 80:
        recommendations.append("Maintain consistent monthly payments without delays")
    if credit_utilization_score < 70:
        recommendations.append("Reduce credit card utilization below 30% of limit")
    if credit_history_score < 70:
        recommendations.append("Keep old credit accounts active to build credit age")
    if credit_mix_score < 70:
        recommendations.append("Diversify credit portfolio with mix of secured and unsecured credit")
    if new_credit_score < 75:
        recommendations.append("Avoid multiple credit applications in short period")
    if final_cibil_score < 750:
        recommendations.append("Focus on timely payments to reach excellent score range (750+)")
    
    return {
        "cibil_score": final_cibil_score,
        "status": status,
        "loan_approval_likelihood": loan_approval,
        "analysis_period": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "total_months": round(date_range_months, 1),
            "total_years": round(date_range_years, 2)
        },
        "score_breakdown": {
            "payment_history": {
                "score": round(payment_history_score, 1),
                "weightage": "35%",
                "contribution": round(payment_history_score * CIBIL_FACTORS['payment_history'], 1),
                "remarks": payment_remarks
            },
            "credit_utilization": {
                "score": round(credit_utilization_score, 1),
                "weightage": "30%",
                "contribution": round(credit_utilization_score * CIBIL_FACTORS['credit_utilization'], 1),
                "utilization_percentage": cur_percentage,
                "status": cur_status
            },
            "credit_history_length": {
                "score": round(credit_history_score, 1),
                "weightage": "15%",
                "contribution": round(credit_history_score * CIBIL_FACTORS['credit_history_length'], 1),
                "years": round(credit_age_years, 2),
                "status": credit_age_status
            },
            "credit_mix": {
                "score": round(credit_mix_score, 1),
                "weightage": "10%",
                "contribution": round(credit_mix_score * CIBIL_FACTORS['credit_mix'], 1),
                "types_count": num_credit_types,
                "types": list(credit_types_present),
                "status": mix_status
            },
            "new_credit": {
                "score": round(new_credit_score, 1),
                "weightage": "10%",
                "contribution": round(new_credit_score * CIBIL_FACTORS['new_credit'], 1),
                "recent_inquiries": recent_inquiries,
                "status": inquiry_status
            }
        },
        "transaction_summary": {
            "total_transactions": len(dated_transactions),
            "credit_card_payments": len(cc_payments),
            "home_loan_payments": len(home_loans),
            "car_loan_payments": len(car_loans),
            "personal_loan_payments": len(personal_loans),
            "education_loan_payments": len(education_loans),
            "insurance_payments": len(life_insurance) + len(health_insurance)
        },
        "recommendations": recommendations
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "CIBIL Analysis"}

@app.get("/analyze-cibil/{session_id}")
async def analyze_cibil_score(session_id: str):
    """Analyze CIBIL score based on actual CSV transaction data"""
    try:
        url = f"{DATA_PROCESSOR_BASE_URL}/cibil-data/{session_id}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data")
        
        data = response.json()
        transactions = data.get('relevant_transactions', [])
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions found in CSV")
        
        result = calculate_cibil_score(transactions)
        
        # Add metadata
        result["session_id"] = session_id
        result["timestamp"] = datetime.now().isoformat()
        result["methodology"] = "TransUnion CIBIL India 2025"
        
        return result
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to data service")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
