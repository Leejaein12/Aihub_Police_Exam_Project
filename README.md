# 코사인 유사도 기반 퀴즈 생성 시스템

## 📌 개요

문장 임베딩과 코사인 유사도를 활용해
정답과 유사하지만 다른 오답을 자동 생성하는 퀴즈 시스템

---

## 🛠 사용 기술

* Flask: 웹 서버
* pandas: CSV 처리
* numpy: 벡터 연산 (코사인 유사도)
* gdown: 구글 드라이브 파일 다운로드

---

## 📥 데이터 다운로드

서버 실행 시 필요한 파일을 다운로드

```python
EMBEDDINGS_ID = "11FDE74NC8wlpzUW-2qex-A4_nuwyEqZL"
CSV_ID = "1GalqZwUXTVSvAlRBaIzscLB3aBcHC-07"

def download_if_missing():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        gdown.download(...)

    if not os.path.exists(EMBEDDINGS_PATH):
        gdown.download(...)
```

---

## 📊 데이터 전처리

```python
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
```

* 결측값 제거
* 인덱스 재정렬

---

## 🧠 임베딩 (Embedding)

문장을 숫자 벡터로 변환한 것

예시:

```
"보안관찰처분대상자의 정의는?"
→ [0.12, -0.34, 0.87, ...] (768차원)
```

```python
embeddings = np.load(EMBEDDINGS_PATH)
```

---

## 📐 코사인 유사도

공식:

```
cos(θ) = A·B / (|A| × |B|)
```

이미 L2 정규화되어 있으므로:

```
cos(θ) = A·B
```

---

## ❌ 오답 생성 로직

### 1. 같은 카테고리에서만 선택

```python
same_cat = df[df["category"] == category].index.tolist()
```

---

### 2. 유사도 계산

```python
query = embeddings[correct_idx].reshape(1, -1)
sims = np.dot(embeddings[same_cat], query.T).flatten()
```

---

### 3. 정답 제외

```python
if correct_idx in same_cat:
    sims[same_cat.index(correct_idx)] = -1
```

---

### 4. 후보 필터링

```
0.3 < 유사도 < 0.8
```

* 0.8 이상 → 너무 비슷
* 0.3 이하 → 너무 다름
* 중간 → 헷갈리는 오답

---

### 5. 오답 선택

```python
if len(candidates) < 3:
    top_indices = np.argsort(sims)[::-1][:3]
    return [df.iloc[same_cat[i]]["answer"] for i in top_indices]

top_pool = candidates[:min(20, len(candidates))]
selected = random.sample(top_pool, min(3, len(top_pool)))

return [df.iloc[i]["answer"] for i, _ in selected]
```

---

## 🎯 퀴즈 생성

```python
@app.route("/quiz")
def get_quiz():
    idx = filtered.sample(1).index[0]
    row = df.iloc[idx]

    wrong_answers = get_wrong_answers(idx, row["category"], n=3)

    choices = wrong_answers + [row["answer"]]
    random.shuffle(choices)

    correct_idx = choices.index(row["answer"])
```

---

## 🔄 전체 흐름

```
서버 시작
→ 데이터 다운로드
→ CSV + 임베딩 로드

사용자 요청
→ 랜덤 문제 선택
→ 코사인 유사도 계산
→ 오답 3개 생성
→ 보기 섞기
→ 반환
```

---

## 💡 핵심 포인트

* 임베딩: 문장의 의미를 숫자로 표현
* 코사인 유사도: 문장 간 의미 유사도 측정
* 적절한 유사도 범위로 자연스러운 오답 생성
