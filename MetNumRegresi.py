import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Fungsi untuk model eksponensial
def exponentialModel(x, a, b):
    return a * np.exp(b * x)

# Fungsi untuk menghitung RMS
def calculateRMS(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculateModel(TB, NT):

    # Model Linear
    linear_coeffs = np.polyfit(TB, NT, 1)
    linear_model = np.poly1d(linear_coeffs)
    NT_linear_pred = linear_model(TB)

    # Model Eksponensial    
    exp_params, _ = curve_fit(exponentialModel, TB, NT, p0=(1, 0.1))
    NT_exponential_pred = exponentialModel(TB, *exp_params)

    # Menghitung galat RMS
    rms_linear = calculateRMS(NT, NT_linear_pred)
    rms_exponential = calculateRMS(NT, NT_exponential_pred)
    
    print(f"Galat RMS Linear: {rms_linear:.4f}\nGalat RMS Eksponensial: {rms_exponential:.4f}")

    # Plot data asli dan hasil regresi linear
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(TB, NT, s=10, label='Data Asli')  
    plt.plot(TB, NT_linear_pred, color='red', label='Regresi Linear')
    plt.xlabel('Durasi Waktu Belajar (jam)')
    plt.ylabel('Nilai Ujian')
    plt.title('Regresi Linear')
    plt.yticks(np.arange(10, 105, 5))
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, f'RMS: {rms_linear:.4f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.subplot(1, 2, 2)
    plt.scatter(TB, NT, s=10, label='Data Asli') 
    plt.plot(TB, NT_exponential_pred, color='green', label='Regresi Eksponensial')
    plt.xlabel('Durasi Waktu Belajar (jam)')
    plt.ylabel('Nilai Ujian')
    plt.title('Regresi Eksponensial')
    plt.yticks(np.arange(10, 105, 5))
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, f'RMS: {rms_exponential:.4f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()

#Note : copy directory dari file "student_performance.csv" jika file disimpan di folder lain
data = pd.read_csv(r"E:\Project VS Code\Project Metnum\student_performance.csv")

# Ekstrak kolom yang diperlukan
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

calculateModel(TB, NT)

