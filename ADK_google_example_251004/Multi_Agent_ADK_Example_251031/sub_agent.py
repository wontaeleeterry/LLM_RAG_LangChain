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
from typing import Optional 
# dict는 Python 내장 타입이므로 임포트하지 않습니다.

# ----------------------------
# 1) Data Loader Sub-Agent
# ----------------------------
# ToolContext를 인수로 받아 세션에 DataFrame을 저장합니다.
def load_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """합성 데이터를 생성하고 'dataframe' 키로 세션에 직렬화 가능한 데이터를 저장합니다. n_samples, n_features 등의 파라미터를 받을 수 있습니다."""
    # params가 None일 때 빈 딕셔너리로 초기화
    if params is None:
        params = {}
    
    n_samples = params.get("n_samples", 300)
    n_features = params.get("n_features", 6)
    n_clusters = params.get("n_clusters", 4)
    random_state = params.get("random_state", 42)

    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=n_samples,
                           n_features=n_features,
                           centers=n_clusters,
                           random_state=random_state)
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(n_features)])
    df['label_true'] = y_true
    rng = np.random.default_rng(random_state)
    # 결측치 3% 생성
    mask = rng.random(df.shape) < 0.03
    df_with_na = df.mask(mask)
    
    # 세션 상태에 직렬화 가능한 형태로 저장 (DataFrame -> list of dict)
    tool_context.state["dataframe"] = df_with_na.to_dict('records') # 👈 직렬화 가능한 형태로 변환
    return f"데이터 생성 완료. 샘플 수: {n_samples}, 결측치 포함."

data_loader_agent = Agent(
    name="data_loader_agent",
    model="gemini-2.0-flash",
    description="데이터 생성 및 로드 전문 에이전트. 'dataframe' 키로 세션에 직렬화 가능한 형태로 저장합니다. n_samples, n_features 등의 파라미터를 받을 수 있습니다.",
    instruction="합성 데이터를 생성하고 결측치를 포함시켜 'dataframe' 키로 세션에 저장합니다.",
    tools=[FunctionTool(load_data_tool)]
)

# ----------------------------
# 2) Data Cleaner Sub-Agent
# ----------------------------
# ToolContext를 통해 'dataframe'을 읽어와 정제 후 다시 저장합니다.
def clean_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """'dataframe'을 세션에서 읽어와 결측치를 처리하고 다시 저장합니다. 추가 파라미터는 없습니다."""
    # 세션에서 데이터를 가져와 DataFrame으로 복원
    state_data = tool_context.state.get("dataframe")
    if state_data is None:
        return "오류: 'dataframe'이 세션에 없습니다. 데이터 로드 에이전트가 먼저 실행되어야 합니다."
    
    df = pd.DataFrame(state_data) # 👈 DataFrame으로 복원
        
    features = [c for c in df.columns if c != 'label_true']
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

    # 세션 상태에 직렬화 가능한 형태로 업데이트 (DataFrame -> list of dict)
    tool_context.state["dataframe"] = df.to_dict('records') # 👈 직렬화 가능한 형태로 변환
    return "데이터 정제 완료. 결측치를 평균값으로 대체했습니다."

data_cleaner_agent = Agent(
    name="data_cleaner_agent",
    model="gemini-2.0-flash",
    description="데이터 정제 전문 에이전트. 'dataframe' 키의 결측치를 처리하고 업데이트합니다.",
    instruction="세션에서 'dataframe'을 가져와 결측치를 처리하고 정제된 데이터를 반환합니다.",
    tools=[FunctionTool(clean_data_tool)]
)

# ----------------------------
# 3) Data Analyzer Sub-Agent
# ----------------------------
# ToolContext를 통해 'dataframe'을 읽어와 분석 후 결과를 저장합니다.
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

    features = [c for c in df.columns if c not in ['label_true', 'cluster', 'pca1', 'pca2']]
    X_scaled = StandardScaler().fit_transform(df[features])
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    k = params.get("n_clusters", 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_pca)
    
    df['cluster'] = cluster_labels
    df['pca1'] = X_pca[:,0]
    df['pca2'] = X_pca[:,1]
    
    # 분석 결과를 세션 상태에 직렬화 가능한 형태로 업데이트 (DataFrame -> list of dict)
    tool_context.state["dataframe"] = df.to_dict('records') # 👈 직렬화 가능한 형태로 변환
    # tool_context.state["analysis_result"] = {"pca": pca, "kmeans": kmeans} # 👈 직렬화 불가능하므로 제거

    return f"데이터 분석 완료. PCA (2차원) 및 KMeans (K={k}) 클러스터링을 수행했습니다. 클러스터 레이블과 PCA 차원을 'dataframe'에 추가했습니다."

data_analyzer_agent = Agent(
    name="data_analyzer_agent",
    model="gemini-2.0-flash",
    description="데이터 분석 전문 에이전트. PCA와 KMeans를 수행하여 분석 결과를 세션에 저장합니다. n_clusters 파라미터를 받을 수 있습니다.",
    instruction="세션에서 'dataframe'을 가져와 PCA와 KMeans를 수행하고 결과를 'dataframe' 키로 세션에 업데이트합니다.",
    tools=[FunctionTool(analyze_data_tool)]
)

# ----------------------------
# 4) Data Visualizer Sub-Agent
# ----------------------------
# ToolContext를 통해 'dataframe'을 읽어와 시각화하고 이미지 경로를 반환합니다.
def visualize_data_tool(tool_context: ToolContext, params: Optional[dict] = None):
    """'dataframe'을 세션에서 읽어와 시각화하고 이미지 경로를 반환합니다. plot_path 파라미터를 받을 수 있습니다."""
    # 세션에서 데이터를 가져와 DataFrame으로 복원
    state_data = tool_context.state.get("dataframe")
    if state_data is None or not isinstance(state_data, list):
         return "오류: 시각화에 필요한 분석 데이터('dataframe' 리스트)가 세션에 없습니다. 분석 에이전트가 먼저 실행되어야 합니다."
         
    df = pd.DataFrame(state_data) # 👈 DataFrame으로 복원
        
    if 'pca1' not in df.columns:
        return "오류: 시각화에 필요한 PCA 차원('pca1', 'pca2')이 데이터프레임에 없습니다. 분석 에이전트가 먼저 실행되어야 합니다."

    # params가 None일 때 빈 딕셔너리로 초기화
    if params is None:
        params = {}
        
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis', s=30, alpha=0.8)
    
    # 범례 추가
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Cluster Visualization (PCA 2D)")
    plt.tight_layout()
    
    out_path = params.get("plot_path", "./cluster_plot.png")
    
    # 실제 환경에서는 파일 시스템 접근 및 저장이 필요합니다.
    # ADK의 파일 저장소(Artifacts) 기능을 사용하거나, ADK Engine에 배포된 환경의 경로를 사용해야 합니다.
    # 로컬 테스트를 위해 임시로 저장합니다.
    try:
        fig.savefig(out_path)
    except Exception as e:
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