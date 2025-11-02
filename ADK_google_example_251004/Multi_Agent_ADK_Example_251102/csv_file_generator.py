import pandas as pd
import numpy as np
import os

def generate_sample_csv(filename: str = 'customer_data.csv', n_samples: int = 500):
    """
    데이터 분석 교육을 위한 가상의 고객 데이터를 생성하고 CSV 파일로 저장합니다.
    (결측치, 범주형/수치형 데이터, 이상치 포함)
    """
    print(f"--- 가상 데이터 생성 시작 ({n_samples}개 샘플) ---")

    # 1. 시드 설정
    np.random.seed(42)

    # 2. 데이터 생성
    
    # 고객 ID (수치형)
    customer_id = np.arange(1000, 1000 + n_samples)
    
    # 나이 (수치형 - 이상치 포함)
    age = np.random.normal(loc=35, scale=10, size=n_samples).astype(int)
    # 5% 확률로 0 이하 또는 100 이상인 이상치 생성
    outlier_mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    age[outlier_mask] = np.random.choice([-10, 105, 120], size=outlier_mask.sum())
    age = np.clip(age, a_min=None, a_max=120) # 최대 120세로 제한
    
    # 성별 (범주형)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.50, 0.02])
    
    # 거주 지역 (범주형)
    regions = ['Seoul', 'Busan', 'Incheon', 'Others']
    region = np.random.choice(regions, size=n_samples, p=[0.40, 0.20, 0.15, 0.25])
    
    # 월 평균 지출 (수치형)
    spending = np.random.lognormal(mean=7.0, sigma=0.8, size=n_samples)
    
    # 구독 여부 (이진 범주형)
    subscription = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    
    # 3. DataFrame 생성
    data = pd.DataFrame({
        'CustomerID': customer_id,
        'Age': age,
        'Gender': gender,
        'Region': region,
        'MonthlySpending': spending.round(2),
        'Subscription': subscription
    })
    
    # 4. 분석 교육을 위해 결측치(NaN) 생성
    
    # Age 컬럼에 결측치 3% 생성
    na_mask_age = np.random.choice([True, False], size=n_samples, p=[0.03, 0.97])
    data.loc[na_mask_age, 'Age'] = np.nan
    
    # MonthlySpending 컬럼에 결측치 2% 생성
    na_mask_spending = np.random.choice([True, False], size=n_samples, p=[0.02, 0.98])
    data.loc[na_mask_spending, 'MonthlySpending'] = np.nan

    # 5. 파일을 CSV로 저장
    data.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"--- 파일 저장 완료: {os.path.abspath(filename)} ---")
    print("\n✅ 데이터 특성:")
    print(f"* 전체 샘플 수: {n_samples}")
    print(f"* 포함된 데이터 타입: 수치형 (Age, MonthlySpending), 범주형 (Gender, Region, Subscription)")
    print(f"* **정제 포인트**: Age와 MonthlySpending에 결측치(NaN)가 포함되어 있습니다.")
    print(f"* **분석 포인트**: Age에 비현실적인 이상치(0 이하, 100 초과)가 포함되어 있습니다.")
    print("\n")

    return filename

# 파일 생성 실행
csv_file_path = generate_sample_csv()