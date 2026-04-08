import json
import os
import pandas as pd

# 경찰 시험 관련 법령 목록
POLICE_LAWS = [
    '형법',
    '형사소송법',
    '형사소송규칙',
    '경범죄 처벌법',
    '경찰공무원법',
    '경찰수사규칙',
    '국가보안법',
    '군형법',
    '도로교통법',
    '범죄수익은닉의 규제 및 처벌 등에 관한 법률',
    '범죄피해자 보호법',
    '보안관찰법',
    '성폭력범죄의 처벌 등에 관한 특례법',
    '아동학대범죄의 처벌 등에 관한 특례법',
    '특정강력범죄의 처벌에 관한 특례법',
    '특정범죄 가중처벌 등에 관한 법률',
    '폭력행위 등 처벌에 관한 법률',
]


def load_law_files(directory):
    """법령 QA - 경찰 시험 관련 법령만 필터링"""
    data = []
    skipped = 0

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue

        with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
            item = json.load(f)

            title = item.get('info', {}).get('title', '')
            if title not in POLICE_LAWS:
                skipped += 1
                continue

            label = item.get('label', {})
            q = label.get('input', '').strip()
            a = label.get('output', '').strip()

            if q and a and len(q) > 10 and len(a) > 20:
                data.append({
                    'question': q,
                    'answer': a,
                    'category': '법령',
                    'law': title
                })

    print(f"  → 필터링 제외: {skipped}개")
    return data


def load_judgment_qa_files(directory):
    """판결문 QA - 전체"""
    data = []

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue

        with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
            item = json.load(f)
            label = item.get('label', {})
            q = label.get('input', '').strip()
            a = label.get('output', '').strip()

            if q and a and len(q) > 10 and len(a) > 20:
                data.append({
                    'question': q,
                    'answer': a,
                    'category': '판결문',
                    'law': '판결문'
                })

    return data


def load_judgment_sum_files(directory):
    """판결문 SUM - 사건명 + 사건번호 → 요약문"""
    data = []

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue

        with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
            item = json.load(f)
            info = item.get('info', {})
            label = item.get('label', {})

            case_name = info.get('caseName', '').strip()
            case_num = info.get('caseNum', '').strip()
            a = label.get('output', '').strip()

            # 문제: "다음 사건의 판결 요약으로 올바른 것은?"
            q = f"다음 사건의 판결 내용으로 올바른 것은? [사건명: {case_name} / 사건번호: {case_num}]"

            if case_name and a and len(a) > 20:
                data.append({
                    'question': q,
                    'answer': a,
                    'category': '요약',
                    'law': '판결문요약'
                })

    return data


def preprocess():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

    LAW_DIR = os.path.join(DATA_DIR, '법령_QA')
    JUDGMENT_QA_DIR = os.path.join(DATA_DIR, '판결문_QA')
    JUDGMENT_SUM_DIR = os.path.join(DATA_DIR, '판결문_SUM')
    OUTPUT_PATH = os.path.join(DATA_DIR, 'quiz_data.csv')

    print("=" * 50)
    print("법령 QA 로딩 중... (경찰 시험 관련만)")
    law_data = load_law_files(LAW_DIR)
    print(f"  → {len(law_data)}개 로드 완료")

    print("판결문 QA 로딩 중...")
    judgment_qa_data = load_judgment_qa_files(JUDGMENT_QA_DIR)
    print(f"  → {len(judgment_qa_data)}개 로드 완료")

    print("판결문 요약 로딩 중...")
    judgment_sum_data = load_judgment_sum_files(JUDGMENT_SUM_DIR)
    print(f"  → {len(judgment_sum_data)}개 로드 완료")
    print("=" * 50)

    df = pd.DataFrame(law_data + judgment_qa_data + judgment_sum_data)
    print(f"\n총 합계: {len(df)}개")
    print("\n카테고리별 분포:")
    print(df['category'].value_counts().to_string())
    print("\n법령별 분포 (법령 카테고리):")
    print(df[df['category'] == '법령']['law'].value_counts().to_string())

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()