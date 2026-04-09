# 코사인 유사도 응용

## 라이브러리 불러오기

- flask → 웹 서버

- pandas → CSV 읽기

- numpy → 행렬 연산 (코사인 유사도에 사용)

- gdown → 구글 드라이브에서 파일 다운로드

---

### 서버 시작할 때 드라이브에서 다운로드
EMBEDDINGS_ID = "11FDE74NC8wlpzUW-2qex-A4_nuwyEqZL"
CSV_ID = "1GalqZwUXTVSvAlRBaIzscLB3aBcHC-07"

---

### 파일이 없을 때만 다운로드하고 이미 있으면 스킵
def download_if_missing():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        gdown.download(CSV_URL, CSV_PATH, quiet=False)

    if not os.path.exists(EMBEDDINGS_PATH):
        gdown.download(EMBEDDINGS_URL, EMBEDDINGS_PATH, quiet=False)


---

### CSV 불러오고 question이나 answer가 비어있는 행 제거.
reset_index로 인덱스 0부터 다시 정렬


df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)

---

### 미리 계산해둔 임베딩 파일 로드
 형태는 (41941, 768) → 41941개 문장, 각각 768차원 벡터

embeddings = np.load(EMBEDDINGS_PATH)

 ---

 ### 임베딩이 무엇인가?
 - "보안관찰처분대상자의 정의는?"

 - → [0.12, -0.34, 0.87, ...] ← 768개 숫자로 변환

 - 문장의 의미를 숫자 벡터로 표현한 것. 의미가 비슷한 문장일수록 벡터 방향이 비슷함

 ---

 ### 오답 추출 함수

 같은 카테고리(법령/판결문/요약) 안에서만 오답 뽑음. 다른 카테고리 섞이면 이상한 보기 나옴

def get_wrong_answers(correct_idx, category, n=3):
    same_cat = df[df['category'] == category].index.tolist()

### 핵심! 코사인 유사도 계산하는 부분
query = embeddings[correct_idx].reshape(1, -1)
sims = np.dot(embeddings[same_cat], query.T).flatten()

- embeddings[correct_idx] → 정답 문장의 벡터 (768차원)

- reshape(1, -1) → 행렬 연산을 위해 shape 변환 (768,) → (1, 768)

- np.dot(embeddings[same_cat], query.T) → 내적(dot product) 계산


---

### 코사인 유사도 원리: cos(θ) = A·B / (|A| × |B|)

- 두 벡터의 내적을 각 벡터의 크기로 나눈 값. 근데 embed.py에서 이미 L2 정규화를 해서 모든 벡터의 크기가 1임. 그래서: cos(θ) = A·B / (1 × 1) = A·B

- 그냥 내적만 하면 코사인 유사도가 나옴

- 결과 sims는 각 문장과 정답 문장의 유사도 점수 배열 -1 ~ 1 사이 값

---

### 정답 자체를 오답으로 뽑으면 안 되니까 정답의 유사도를 -1로 설정해서 제외

if correct_idx in same_cat:
    correct_pos = same_cat.index(correct_idx)
    sims[correct_pos] = -1


---

### 유사도가 0.3~0.8 사이인 것만 후보로 선택
candidates = [(same_cat[i], s) for i, s in enumerate(sims) if 0.3 < s < 0.8]
candidates.sort(key=lambda x: x[1], reverse=True)


   - 0.8 이상 → 너무 비슷 → 정답이랑 헷갈려서 오답 판별 어려움

   - 0.3 이하 → 너무 다름 → 보기가 엉뚱해짐

   - 0.3~0.8 → 적당히 비슷 → 헷갈리는 오답

### 후보가 3개 미만이면 그냥 유사도 높은 순으로 뽑고, 충분하면 상위 20개 중에서 랜덤으로 3개 선택. 매번 같은 보기 나오는 거 방지

if len(candidates) < n:
    top_indices = np.argsort(sims)[::-1][:n]
    return [df.iloc[same_cat[i]]['answer'] for i in top_indices]

top_pool = candidates[:min(20, len(candidates))]
selected = random.sample(top_pool, min(n, len(top_pool)))
return [df.iloc[i]['answer'] for i, _ in selected]

---

### 랜덤 문제 1개 뽑고 → 오답 3개 생성 → 정답 1개 합쳐서 4개 → 섞기 → 정답 위치 기록

@app.route('/quiz')
def get_quiz():
    idx = filtered.sample(1).index[0]
    row = df.iloc[idx]

    wrong_answers = get_wrong_answers(idx, row['category'], n=3)

    choices = wrong_answers + [row['answer']]
    random.shuffle(choices)

    correct_idx = choices.index(row['answer'])

---

## 전체 흐름 요약

- 서버 시작

- → 구글 드라이브에서 CSV + embeddings.npy 다운로드

- → 메모리에 로드

- 사용자 요청

- → 랜덤 문제 선택

- → 정답 벡터와 다른 문장들 코사인 유사도 계산

- → 적당히 비슷한 것 3개 오답으로 선택

- → 보기 4개 섞어서 반환