from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools import ToolContext
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any 

# ----------------------------
# 1) Data Loader Sub-Agent (수정됨)
# ----------------------------
def load_data_tool(tool_context: ToolContext, params: Dict[str, Any]):
    """
    제공된 'file_path'에 있는 CSV 파일을 Pandas DataFrame으로 로드하고 
    'dataframe' 키로 세션에 직렬화 가능한 데이터를 저장합니다.
    
    필수 파라미터:
    - 'file_path': 로드할 CSV 파일의 경로 (예: './data.csv')
    """
    # 필수 파라미터인 'file_path' 확인
    file_path = params.get("file_path")
    if not file_path:
        return "오류: CSV 파일을 로드하기 위한 'file_path' 파라미터가 필요합니다."
    
    try:
        # ⚠️ 핵심 변경: 사용자 CSV 파일 로드
        df = pd.read_csv(file_path)
        
        # 세션 상태에 직렬화 가능한 형태로 저장 (DataFrame -> list of dict)
        tool_context.state["dataframe"] = df.to_dict('records') # 👈 직렬화 가능한 형태로 변환
        
        # 로드된 데이터 정보 반환
        return (f"데이터 로드 완료. 파일 경로: {file_path}, 샘플 수: {len(df)}, "
                f"특징 수: {len(df.columns)}. 데이터 분석을 시작하세요.")
        
    except FileNotFoundError:
        return f"오류: 지정된 파일 경로에 CSV 파일이 없습니다: {file_path}"
    except pd.errors.EmptyDataError:
        return f"오류: 파일이 비어 있습니다: {file_path}"
    except Exception as e:
        return f"데이터 로드 중 오류 발생: {e}"

data_loader_agent = Agent(
    name="data_loader_agent",
    model="gemini-2.0-flash",
    # ⚠️ 설명 변경
    description="사용자 입력에서 CSV 파일 경로를 받아 데이터를 로드하고 'dataframe' 키로 세션에 저장하는 전문 에이전트. 'file_path' 파라미터가 필요합니다.",
    # ⚠️ 명령 변경
    instruction="사용자에게 받은 'file_path'를 사용하여 CSV 파일을 로드하고 'dataframe' 키로 세션에 저장합니다. 파일 경로가 파라미터에 포함되어 있는지 확인하세요.",
    tools=[FunctionTool(load_data_tool)]
)

# ----------------------------
# 2) Data Cleaner Sub-Agent (기존 코드 유지)
# ----------------------------
def clean_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """'dataframe'을 세션에서 읽어와 결측치를 처리하고 다시 저장합니다. 추가 파라미터는 없습니다."""
    # 세션에서 데이터를 가져와 DataFrame으로 복원
    state_data = tool_context.state.get("dataframe")
    if state_data is None:
        return "오류: 'dataframe'이 세션에 없습니다. 데이터 로드 에이전트가 먼저 실행되어야 합니다."
    
    df = pd.DataFrame(state_data) # 👈 DataFrame으로 복원
        
    # 'label_true'는 이전 합성 데이터 코드에서 사용되었으나, 새로운 CSV에서는 없을 수 있습니다.
    # 안전하게 숫자형 컬럼만 선택하여 결측치 처리
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_features:
        return "경고: 정제할 숫자형 데이터가 없습니다. 문자열 컬럼은 정제하지 않고 다음 단계로 넘어갑니다."
        
    imputer = SimpleImputer(strategy='mean')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])

    # 세션 상태에 직렬화 가능한 형태로 업데이트 (DataFrame -> list of dict)
    tool_context.state["dataframe"] = df.to_dict('records') # 👈 직렬화 가능한 형태로 변환
    return "데이터 정제 완료. 숫자형 컬럼의 결측치를 평균값으로 대체했습니다."

data_cleaner_agent = Agent(
    name="data_cleaner_agent",
    model="gemini-2.0-flash",
    description="데이터 정제 전문 에이전트. 'dataframe' 키의 결측치를 처리하고 업데이트합니다.",
    instruction="세션에서 'dataframe'을 가져와 숫자형 데이터의 결측치를 처리하고 정제된 데이터를 반환합니다.",
    tools=[FunctionTool(clean_data_tool)]
)

# ----------------------------
# 3) Data Analyzer Sub-Agent (수정됨 - 분석 컬럼 명시)
# ----------------------------
def analyze_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """'dataframe'을 세션에서 읽어와 PCA와 KMeans를 수행하고 결과를 세션에 직렬화 가능한 형태로 저장합니다. n_clusters 파라미터를 받을 수 있습니다."""
    # 세션에서 데이터를 가져와 DataFrame으로 복원
    state_data = tool_context.state.get("dataframe")
    if state_data is None:
        return "오류: 'dataframe'이 세션에 없습니다. 데이터 로드/정제 에이전트가 먼저 실행되어야 합니다."
        
    df = pd.DataFrame(state_data) # 👈 DataFrame으로 복원
        
    # params가 None일 때 빈 딕셔너리로 초기화
    if params is None:
        params = {}

    # ⚠️ 수정된 핵심 로직: 분석에 사용할 컬럼을 명시 (customer_data.csv 기준)
    analysis_features = ['Age', 'MonthlySpending']
    
    # 데이터프레임에 해당 컬럼이 있는지 확인 (강건성 강화)
    X = df[[col for col in analysis_features if col in df.columns]]
    
    # 데이터 타입 확인 및 숫자형으로 변환 시도 (cleaner에서 처리 안된 이상치/문자열이 있을 경우)
    X = X.apply(pd.to_numeric, errors='coerce').dropna()

    if X.empty or len(X.columns) < 2:
        return "오류: PCA/KMeans를 수행할 수 있는 최소 2개 이상의 숫자형 특징이 데이터프레임에 없거나 모두 결측치입니다. 데이터 정제 단계를 확인하세요."

    # 2. 스케일링
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. PCA 및 KMeans 수행
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    k = params.get("n_clusters", 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # 4. 결과 업데이트
    # 결과를 원본 DataFrame에 병합
    # X에 대한 인덱스를 사용하여 원본 df에 추가
    df.loc[X.index, 'cluster'] = cluster_labels
    df.loc[X.index, 'pca1'] = X_pca[:,0]
    df.loc[X.index, 'pca2'] = X_pca[:,1]
    
    # 분석 결과를 세션 상태에 직렬화 가능한 형태로 업데이트
    tool_context.state["dataframe"] = df.to_dict('records')

    return f"데이터 분석 완료. PCA (2차원) 및 KMeans (K={k}) 클러스터링을 수행했습니다. 클러스터 레이블과 PCA 차원을 'dataframe'에 추가했습니다."

data_analyzer_agent = Agent(
    name="data_analyzer_agent",
    model="gemini-2.0-flash",
    description="데이터 분석 전문 에이전트. PCA와 KMeans를 수행하여 분석 결과를 세션에 저장합니다. n_clusters 파라미터를 받을 수 있습니다.",
    instruction="세션에서 'dataframe'을 가져와 PCA와 KMeans를 수행하고 결과를 'dataframe' 키로 세션에 업데이트합니다.",
    tools=[FunctionTool(analyze_data_tool)]
)

# ----------------------------
# 4) Data Visualizer Sub-Agent (기존 코드 유지)
# ----------------------------
def visualize_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """'dataframe'을 세션에서 읽어와 시각화하고 이미지 경로를 반환합니다. plot_path 파라미터를 받을 수 있습니다."""
    # 세션에서 데이터를 가져와 DataFrame으로 복원
    state_data = tool_context.state.get("dataframe")
    if state_data is None or not isinstance(state_data, list):
         return "오류: 시각화에 필요한 분석 데이터('dataframe' 리스트)가 세션에 없습니다. 분석 에이전트가 먼저 실행되어야 합니다."
         
    df = pd.DataFrame(state_data) # 👈 DataFrame으로 복원
        
    if 'pca1' not in df.columns or 'cluster' not in df.columns:
        return "오류: 시각화에 필요한 분석 결과('pca1', 'pca2', 'cluster')가 데이터프레임에 없습니다. 분석 에이전트가 먼저 실행되어야 합니다."

    # params가 None일 때 빈 딕셔너리로 초기화
    if params is None:
        params = {}
        
    # PCA 결과와 클러스터가 있는 행만 시각화
    plot_df = df.dropna(subset=['pca1', 'pca2', 'cluster'])
    
    if plot_df.empty:
        return "오류: 시각화할 데이터가 없습니다. 분석 단계에서 모든 데이터가 제거되었을 수 있습니다."
        
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(plot_df['pca1'], plot_df['pca2'], 
                         c=plot_df['cluster'].astype(int), # 클러스터 레이블은 정수형으로 변환
                         cmap='viridis', s=30, alpha=0.8)
    
    # 범례 추가
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Cluster Visualization (PCA 2D)")
    plt.tight_layout()
    
    out_path = params.get("plot_path", "./cluster_plot.png")
    
    # 파일 저장 (기존 로직 유지)
    try:
        fig.savefig(out_path)
    except Exception as e:
        # ADK 환경에서는 Artifacts를 사용하는 것이 일반적입니다.
        return f"경고: 파일 저장 중 오류 발생 ({e}). ADK Engine 환경에서는 Artifacts를 사용해야 합니다. 임시 경로: {out_path}"

    plt.close(fig)
    return out_path

data_visualizer_agent = Agent(
    name="data_visualizer_agent",
    model="gemini-2.0-flash",
    description="데이터 시각화 전문 에이전트. 분석 결과를 시각화하여 이미지 경로를 반환합니다. plot_path 파라미터를 받을 수 있습니다.",
    instruction="세션에서 분석된 'dataframe'을 가져와 클러스터 시각화를 수행하고 이미지 경로를 반환합니다.",
    tools=[FunctionTool(visualize_data_tool)]
)