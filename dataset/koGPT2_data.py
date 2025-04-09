import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# KoGPT2 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPU 사용 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 피싱 및 정상 대화 프롬프트 예제
phishing_prompts = [
    "안녕하세요. 고객님, 은행 계좌가 위험합니다. 지금 바로",
    "고객님, 긴급 공지입니다. 카드 정보가 유출되어",
    "본인 인증이 필요합니다. 휴대폰 번호와 계좌번호를 입력해 주세요."
]
normal_prompts = [
    "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "최근에 읽은 책 중에서 어떤 게 가장 기억에 남으세요?",
    "이번 주말에 가족들과 여행을 가려고 합니다."
]

# 데이터 저장 리스트
data = []


# 데이터 생성 함수
def generate_text(prompt, label, id_num, max_length=500):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"id": id_num, "transcript": generated_text, "label": label}


# 데이터 생성 개수
num_samples = 1000
start_time = time.time()

for i in range(num_samples):
    prompt = phishing_prompts[i % len(phishing_prompts)] if i % 2 == 0 else normal_prompts[i % len(normal_prompts)]
    label = 1 if i % 2 == 0 else 0  # 피싱(1), 정상(0)

    conversation = generate_text(prompt, label, i + 1)
    data.append(conversation)

# 데이터프레임으로 변환
df = pd.DataFrame(data)

# CSV 파일 저장
csv_filename = "phishing_conversations_500tokens.csv"
df.to_csv(csv_filename, index=False, encoding="utf-8")

end_time = time.time()
total_time = round(end_time - start_time, 2)

print(f"✅ {num_samples}개 데이터 생성 완료! 파일 저장: {csv_filename}")
print(f"⏱️ 총 소요 시간: {total_time}초 ({total_time / 60:.2f}분)")
