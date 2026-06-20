import pandas as pd
from datetime import datetime
import random

USER_COUNT=600
USER_PAYMENTS=5000

random.seed(42)
users_data=[]

for i in range(USER_COUNT):
    id=i
    age=random.randint(18, 70)
    date=datetime(2025,random.randint(1, 12), random.randint(1, 28))
    users_data.append({
        "user_id": id,
        "age": age,
        "registration_date": date
    })

payments=[]
for i in range(USER_PAYMENTS):
    user_id = random.randint(0,USER_COUNT-1)
    amount=random.randint(10,100)
    month=users_data[user_id]["registration_date"].month
    date=datetime(2025,random.randint(month,12),random.randint(1,28))
    payments.append({
        "user_id":user_id,
        "amount":amount,
        "payment_date":date
    })

activities=[]
for i in range(USER_COUNT-1):
    num_activ=random.randint(5,20)
    for k in range(num_activ):
        random_year=random.randint(2025,2026)
        activity_date=0
        if random_year==2025:
            activity_date=datetime(2025,random.randint(users_data[i]["registration_date"].month,12),random.randint(1,28))
        else:
            activity_date=datetime(2026,random.randint(1,12),random.randint(1,28))   
        duration_minutes=random.randint(1,120)
        activities.append({
            "user_id":i,
            "activity_date":activity_date,
            "duration_minutes":duration_minutes
        })

pd.DataFrame(users_data).to_csv('data/raw/users.csv', index=False)
pd.DataFrame(payments).to_csv('data/raw/payments.csv', index=False)
pd.DataFrame(activities).to_csv('data/raw/activity.csv', index=False)
