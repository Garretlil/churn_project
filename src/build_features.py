import pandas as pd

df_users = pd.read_csv("data/raw/users.csv")
df_payments = pd.read_csv("data/raw/payments.csv")
df_activity = pd.read_csv("data/raw/activity.csv")

df_payments_ft = (
    df_payments.groupby('user_id')
    .agg(
        total_payment_amount=('amount', 'sum'),
        total_payments_count=('amount', 'count'),
        avg_payment_amount=('amount', 'mean')
    )
    .reset_index()
)

df_activity_ft = (
    df_activity.groupby('user_id')
    .agg(
        last_activity_date=('activity_date', 'max'),
        total_activity_duration=('duration_minutes', 'sum'),
        avg_activity_duration=('duration_minutes', 'mean')
    )
    .reset_index()
)

dt = pd.Timestamp.now().normalize()
df_activity_ft['last_activity_date'] = pd.to_datetime(df_activity_ft['last_activity_date'])
df_activity_ft['days_since_last_activity'] = (dt - df_activity_ft['last_activity_date']).dt.days
df_activity_ft = df_activity_ft.drop(columns=['last_activity_date'])

df_users['days_as_customer'] = (dt - pd.to_datetime(df_users['registration_date'])).dt.days
df_users = df_users.drop(columns=['registration_date'])

df = df_users.merge(df_payments_ft, on='user_id', how='left')
df = df.merge(df_activity_ft, on='user_id', how='left')

fill_values = {
    'total_payment_amount': 0,
    'total_payments_count': 0,
    'avg_payment_amount': 0,
    'days_since_last_activity': 999,
    'total_activity_duration': 0,
    'avg_activity_duration': 0
}
df = df.fillna(fill_values)

df['label'] = (df['days_since_last_activity'] > 45).astype(int)

df.to_parquet('data/features/features.parquet', index=False)
