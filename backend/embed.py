import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
 
def embed():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
 
    CSV_PATH = os.path.join(DATA_DIR, 'quiz_data.csv')
    EMB_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
 
    # ─── 데이터 로드 ───
    print("데이터 로딩 중...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)
    print(f"  → {len(df)}개 로드 완료")
 
    # ─── 모델 로드 ───
    print("모델 로딩 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
 
    # ─── 임베딩 계산 ───
    print("임베딩 계산 중... (시간이 걸려요)")
    embeddings = model.encode(
        df['answer'].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
 
    # ─── L2 정규화 ───
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings = embeddings / norms
 
    # ─── 저장 ───
    np.save(EMB_PATH, embeddings)
    print(f"\n✅ 저장 완료 → {EMB_PATH}")
    print(f"   shape: {embeddings.shape}")
 
if __name__ == "__main__":
    embed()
 