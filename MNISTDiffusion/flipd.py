import torch
from torchvision.utils import save_image
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import math
import matplotlib.pyplot as plt
import os


ckpt_path = "results/steps_00046900.pt"   # ✅ 저장된 체크포인트 경로
n_samples = 1                             # ✅ 생성할 샘플 개수
model_base_dim = 64                       # ✅ 학습할 때 사용했던 base dim
timesteps = 1000                          # ✅ 학습할 때 사용했던 diffusion steps
no_clip = False                           # ✅ x_0 clipping 쓸지 여부
use_cpu = False                           # ✅ True면 CPU 사용, False면 GPU 사용
device = "cpu" if use_cpu else "cuda"

def load_model(ckpt_path, device, image_size=28, in_channels=1, model_base_dim=64, timesteps=1000):
    model = MNISTDiffusion(
        timesteps=timesteps,
        image_size=image_size,
        in_channels=in_channels,
        base_dim=model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    model_ema = ExponentialMovingAverage(model, device=device, decay=0.0)  # decay는 중요 X (로드할 거니까)

    # checkpoint 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    return model_ema

# 모델 로드
model_ema = load_model(
    ckpt_path=ckpt_path,
    device=device,
    image_size=28,
    in_channels=1,
    model_base_dim=model_base_dim,
    timesteps=timesteps
)

model_ema.eval()

# 샘플 생성
samples, flipd = model_ema.module.sampling(
    n_samples=n_samples,
    clipped_reverse_diffusion=not no_clip,
    device=device
)

# # 결과 저장
# os.makedirs("inference_results", exist_ok=True)
# save_image(samples, "inference_results/generated_samples.png", nrow=int(math.sqrt(n_samples)))

# print(f"✅ 샘플 이미지를 'inference_results/generated_samples.png'로 저장 완료했습니다!")


# 저장할 폴더 경로
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

# 그래프 그리기
plt.plot(flipd)
plt.title("Line Plot of List")
plt.xlabel("Index")
plt.ylabel("Value")
plt.xlim(0, 1)
plt.grid(True)

# 저장하기 (예: figures/flipd_plot.png)
output_path = os.path.join(output_dir, "flipd_plot.png")
plt.savefig(output_path)