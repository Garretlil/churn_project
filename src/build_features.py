import pandas as pd
import numpy as np

df_users=pd.read_csv("data/raw/users.csv")
df_payments=pd.read_csv("data/raw/payments.csv")
df_activity=pd.read_csv("data/raw/activity.csv")

df_merged=pd.merge(df_users,df_payments,on="user_id",how="inner")
df_payments_ft = (
    df_merged.groupby('user_id').agg(
        total_payment_amount=('amount','sum'),
        total_payments_count=('amount','count'),
        avg_payment_amount=('amount','mean')
    ).reset_index()
)
df_merged=pd.merge(df_users,df_activity,on='user_id',how='inner')
df_activity_ft=( df_merged.groupby('user_id').agg(
    days_since_last_activity=('activity_date','max'),
    total_activity_duration=('duration_minutes','sum'),
    avg_activity_duration=('duration_minutes','mean')
    ).reset_index()
)
df_activity_ft['days_since_last_activity'] = pd.to_datetime(df_activity_ft['days_since_last_activity'])
dt=pd.Timestamp.now().normalize()
df_activity_ft['days_since_last_activity']=(dt-df_activity_ft['days_since_last_activity']).dt.days


df_users['days_as_customer']=(dt-pd.to_datetime(df_users['registration_date'])).dt.days
df_users=df_users.drop(columns='registration_date')

df=pd.merge(df_users,df_payments_ft,on="user_id",how="left")
df=pd.merge(df,df_activity_ft,on="user_id",how="left")
print(df)
df['total_payment_amount'] = df['total_payment_amount'].fillna(0)
df['total_payments_count'] = df['total_payments_count'].fillna(0)
df['avg_payment_amount'] = df['avg_payment_amount'].fillna(0)
df['days_since_last_activity'] = df['days_since_last_activity'].fillna(999)
df['total_activity_duration'] = df['total_activity_duration'].fillna(0)
df['avg_activity_duration'] = df['avg_activity_duration'].fillna(0)

cond=[
    (df['days_since_last_activity']>30) & (df['total_payments_count']<15),
    (df['days_since_last_activity']<30) & (df['total_payments_count']>=2),
]

choice=[1,0]
df['label']=np.select(cond,choice,default=0)
print((df['days_since_last_activity']<=1).sum())
print((df['label']==0).sum())

pd.DataFrame(df).to_parquet('data/features/features.parquet',index=False)

