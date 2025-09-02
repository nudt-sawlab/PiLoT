import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class PASTA:
    def __init__(self, alpha: float, beta: float, k: int):
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        fft_src = torch.fft.fftn(img, dim=[-2, -1])
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        X, Y = amp_src.shape[1:]
        X_range = np.fft.fftfreq(X, d=1.0)
        Y_range = np.fft.fftfreq(Y, d=1.0)
        XX, YY = np.meshgrid(Y_range, X_range)

        inv = np.sqrt(XX ** 2 + YY ** 2)
        inv = inv / inv.max() * self.alpha
        inv = np.power(inv, self.k) + self.beta
        prop = np.fft.fftshift(inv)
        prop = np.tile(prop[None, :, :], (3, 1, 1))  # repeat for 3 channels

        amp_src *= torch.tensor(np.random.normal(1.0, prop), dtype=amp_src.dtype)

        aug_img = amp_src * torch.exp(1j * pha_src)
        aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
        aug_img = torch.real(aug_img)
        return torch.clamp(aug_img, 0, 1)


def compute_log_spectrum(tensor_img):
    gray = tensor_img.mean(dim=0).numpy()
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude


def main(image_path):
    # 读取图像并转为tensor
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb / 255., dtype=torch.float32).permute(2, 0, 1)

    # 初始化增强器
    pasta = PASTA(alpha=3.0, beta=0.25, k=2)
    img_aug = pasta(img_tensor)

    # 频谱
    spectrum_orig = compute_log_spectrum(img_tensor)
    spectrum_aug = compute_log_spectrum(img_aug)

    # 显示
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(spectrum_orig, cmap="magma")
    axs[0, 1].set_title("Original Spectrum")
    axs[1, 0].imshow(img_aug.permute(1, 2, 0).numpy())
    axs[1, 0].set_title("Augmented Image")
    axs[1, 1].imshow(spectrum_aug, cmap="magma")
    axs[1, 1].set_title("Augmented Spectrum")
    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("/mnt/sda/ycb/0_0/0_0.png")  # 替换为你自己的图像路径
