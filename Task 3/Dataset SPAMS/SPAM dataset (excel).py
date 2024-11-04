import pandas as pd

# Memuat dataset dari file CSV
sms_data = pd.read_csv("spam.csv", encoding='latin-1')  # Pastikan file CSV ada di direktori yang sama atau berikan path lengkap

# Menyimpan DataFrame ke file Excel
sms_data.to_excel("spam_dataset.xlsx", index=False, engine='openpyxl')  # Menyimpan sebagai spam_dataset.xlsx

print("File berhasil disimpan sebagai 'spam_dataset.xlsx'")
