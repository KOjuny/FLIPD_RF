import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
from easydict import EasyDict  # easydict import 추가

def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
    ])

    train_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root="./mnist_data", train=False, download=True, transform=preprocess)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    )

def main(args):
    device = "cpu" if args.cpu else "cuda"
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)
    
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy='cos'
    )
    loss_fn = nn.MSELoss(reduction='mean')

    # Load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    for epoch in range(args.epochs):
        model.train()
        for step, (image, target) in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)

            global_steps += 1

            if step % args.log_freq == 0:
                print(
                    f"Epoch[{epoch+1}/{args.epochs}], "
                    f"Step[{step}/{len(train_dataloader)}], "
                    f"loss: {loss.detach().cpu().item():.5f}, "
                    f"lr: {scheduler.get_last_lr()[0]:.5f}"
                )

        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }
        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, f"results/steps_{global_steps:08d}.pt")

        model_ema.eval()
        samples = model_ema.module.sampling(
            args.n_samples,
            clipped_reverse_diffusion=not args.no_clip,
            device=device
        )
        save_image(samples, f"results/steps_{global_steps:08d}.png", nrow=int(math.sqrt(args.n_samples)))

if __name__ == "__main__":
    args = EasyDict({
        'lr': 0.001,
        'batch_size': 128,
        'epochs': 100,
        'ckpt': '',  # 체크포인트 경로 없으면 빈 문자열
        'n_samples': 36,
        'model_base_dim': 64,
        'timesteps': 1000,
        'model_ema_steps': 10,
        'model_ema_decay': 0.995,
        'log_freq': 10,
        'no_clip': False,
        'cpu': False,
    })
    
    main(args)
