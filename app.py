from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import random
import os
import gdown

app = Flask(__name__)

# ─── 구글 드라이브 파일 ID ───
EMBEDDINGS_ID = "11FDE74NC8wlpzUW-2qex-A4_nuwyEqZL"
CSV_ID = "1GalqZwUXTVSvAlRBaIzscLB3aBcHC-07"

EMBEDDINGS_PATH = "data/embeddings.npy"
CSV_PATH = "data/quiz_data.csv"

def download_if_missing():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print("quiz_data.csv 다운로드 중...")
        gdown.download(f"https://drive.google.com/uc?id={CSV_ID}", CSV_PATH, quiet=False)
        print("✅ quiz_data.csv 다운로드 완료!")

    if not os.path.exists(EMBEDDINGS_PATH):
        print("embeddings.npy 다운로드 중...")
        gdown.download(f"https://drive.google.com/uc?id={EMBEDDINGS_ID}", EMBEDDINGS_PATH, quiet=False)
        print("✅ embeddings.npy 다운로드 완료!")

print("데이터 로딩 중...")
download_if_missing()

# ─── 데이터 로드 ───
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)

# ─── 임베딩 로드 ───
embeddings = np.load(EMBEDDINGS_PATH)

print(f"✅ 준비 완료! 총 {len(df)}개 문제")


def get_wrong_answers(correct_idx, category, n=3):
    """같은 카테고리 내에서 유사도 기반 오답 추출"""
    same_cat = df[df['category'] == category].index.tolist()

    query = embeddings[correct_idx].reshape(1, -1)
    sims = np.dot(embeddings[same_cat], query.T).flatten()

    if correct_idx in same_cat:
        correct_pos = same_cat.index(correct_idx)
        sims[correct_pos] = -1

    candidates = [(same_cat[i], s) for i, s in enumerate(sims) if 0.3 < s < 0.8]
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) < n:
        top_indices = np.argsort(sims)[::-1][:n]
        return [df.iloc[same_cat[i]]['answer'] for i in top_indices]

    top_pool = candidates[:min(20, len(candidates))]
    selected = random.sample(top_pool, min(n, len(top_pool)))
    return [df.iloc[i]['answer'] for i, _ in selected]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/quiz')
def get_quiz():
    category = request.args.get('category', '전체')

    if category == '전체':
        filtered = df
    else:
        filtered = df[df['category'] == category]

    if len(filtered) == 0:
        return jsonify({'error': '해당 카테고리 문제가 없습니다.'}), 404

    idx = filtered.sample(1).index[0]
    row = df.iloc[idx]

    wrong_answers = get_wrong_answers(idx, row['category'], n=3)
    choices = wrong_answers + [row['answer']]
    random.shuffle(choices)
    correct_idx = choices.index(row['answer'])

    return jsonify({
        'question': row['question'],
        'choices': choices,
        'correct_idx': correct_idx,
        'category': row['category'],
        'law': row.get('law', '')
    })


@app.route('/stats')
def get_stats():
    stats = df['category'].value_counts().to_dict()
    return jsonify(stats)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)