import torch
from torchvision.utils import save_image
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import math
import matplotlib.pyplot as plt
import numpy as np
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

for i in range(1):
    # 샘플 생성
    samples, flipd, timestep = model_ema.module.sampling(
        n_samples=n_samples,
        clipped_reverse_diffusion=no_clip,
        device=device
    )

    # 파일 인덱스 문자열 생성 (예: 001)
    index_str = str(i + 1).zfill(3)

    # 🔹 이미지 저장
    image_path = f"results/images/generated_samples_{index_str}.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    save_image(samples, image_path, nrow=int(math.sqrt(n_samples)))
    print(f"✅ 샘플 이미지를 '{image_path}'로 저장 완료했습니다!")

    # 🔹 FLIPD 그래프 저장
    fig_path = f"results/figures/flipd_plot_{index_str}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure()
    plt.plot(timestep, flipd, label="FLIPD")

    # # 🔸 최소값 계산 및 표시
    # min_idx = int(np.argmin(flipd))
    # min_val = flipd[min_idx]
    # plt.scatter(min_idx, min_val, color='red', zorder=5)
    # plt.annotate(f"min={min_val:.4f}", xy=(min_idx, min_val),
    #              xytext=(min_idx + 2, min_val),
    #              arrowprops=dict(facecolor='red', shrink=0.05),
    #              fontsize=9, color='red')

    # 그래프 설정
    plt.title("Line Plot of FLIPD")
    plt.xlabel("Timestep")
    plt.ylabel("FLIPD Value")
    plt.grid(True)
    plt.legend()
    plt.xlim(0,1)
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ FLIPD 그래프를 '{fig_path}'로 저장 완료했습니다!")

# # 샘플 생성
# samples, flipd = model_ema.module.sampling(
#     n_samples=n_samples,
#     clipped_reverse_diffusion= not no_clip,
#     device=device
# )
# flipd = flipd[:-1]
# # 결과 저장
# os.makedirs("results/images", exist_ok=True)
# save_image(samples, "results/images/generated_samples.png", nrow=int(math.sqrt(n_samples)))

# print(f"✅ 샘플 이미지를 'results/images/generated_samples.png'로 저장 완료했습니다!")


# # 저장할 폴더 경로
# output_dir = "results/figures"
# os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

# # 그래프 그리기
# plt.plot(flipd)
# plt.title("Line Plot of List")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.xlim()
# plt.grid(True)

# # 저장하기 (예: figures/flipd_plot.png)
# output_path = os.path.join(output_dir, "flipd_plot.png")
# plt.savefig(output_path)