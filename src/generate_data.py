import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

USER_COUNT = 600
PAYMENTS_PER_USER_MIN = 2
PAYMENTS_PER_USER_MAX = 20
ACTIVITY_PER_USER_MIN = 5
ACTIVITY_PER_USER_MAX = 25

np.random.seed(42)
random.seed(42)

users_data = []
for i in range(USER_COUNT):
    age = np.random.randint(18, 70)
    registration_date = datetime(2025, np.random.randint(1, 12), np.random.randint(1, 28))
    
    income_base = 30 + age * 0.8 + np.random.randn() * 15
    income = max(15, min(120, income_base)) 
    
    spend_per_payment = 50 + income * 1.5 + np.random.randn() * 30
    spend_per_payment = max(15, min(500, spend_per_payment))
    
    users_data.append({
        "user_id": i,
        "age": age,
        "registration_date": registration_date,
        "income": int(income),
        "spend_per_payment": int(spend_per_payment)
    })

payments = []
for user in users_data:
    user_id = user["user_id"]
    reg_date = user["registration_date"]
    spend = user["spend_per_payment"]
    
    n_payments = np.random.randint(PAYMENTS_PER_USER_MIN, PAYMENTS_PER_USER_MAX)
    
    for _ in range(n_payments):
        if np.random.random() < 0.8:
            days_offset = np.random.randint(1, 90)
        else:
            days_offset = np.random.randint(90, 365)
        
        payment_date = reg_date + timedelta(days=days_offset)
        if payment_date > datetime(2025, 12, 31):
            payment_date = datetime(2025, 12, 28)
        
        amount = int(spend * (0.7 + np.random.rand() * 0.6))
        amount = max(10, min(500, amount))
        
        payments.append({
            "user_id": user_id,
            "amount": amount,
            "payment_date": payment_date
        })

activities = []
for user in users_data:
    user_id = user["user_id"]
    reg_date = user["registration_date"]
    
    n_activities = np.random.randint(ACTIVITY_PER_USER_MIN, ACTIVITY_PER_USER_MAX)
    
    for _ in range(n_activities):
        if np.random.random() < 0.6:
            days_offset = np.random.randint(1, 60)
        else:
            days_offset = np.random.randint(60, 365)
        
        activity_date = reg_date + timedelta(days=days_offset)
        if activity_date > datetime(2026, 6, 1):  
            activity_date = datetime(2026, np.random.randint(1, 6), np.random.randint(1, 28))
        
        duration = int(np.random.lognormal(mean=2.5, sigma=0.8))
        duration = max(1, min(180, duration))
        
        activities.append({
            "user_id": user_id,
            "activity_date": activity_date,
            "duration_minutes": duration
        })

pd.DataFrame(users_data).to_csv('data/raw/users.csv', index=False)
pd.DataFrame(payments).to_csv('data/raw/payments.csv', index=False)
pd.DataFrame(activities).to_csv('data/raw/activity.csv', index=False)
