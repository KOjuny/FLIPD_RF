import torch
import math
import os

from torchvision.utils import save_image
from utils_DDPM import DDPM, generate_new_images
from utils_UNet import UNet

# 설정
store_path = "weights/ddpm_fashion.pt"  # 저장된 학습된 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DDPM 하이퍼파라미터
n_steps = 1000
min_beta = 1e-4
max_beta = 0.02
n_samples = 100

# 모델 초기화 및 파라미터 로드
model = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
model.load_state_dict(torch.load(store_path, map_location=device))
model.eval()
print("✅ 학습된 모델 로드 완료")

# 이미지 생성
print("🧨 새로운 이미지 생성 중...")
samples = generate_new_images(model, n_samples=n_samples, device=device, gif_name="fashion.gif")

# 결과 저장
image_path = f"results/images/generated_samples.png"
os.makedirs(os.path.dirname(image_path), exist_ok=True)
save_image(samples, image_path, nrow=int(math.sqrt(n_samples)))
print(f"✅ 샘플 이미지를 '{image_path}'로 저장 완료했습니다!")
