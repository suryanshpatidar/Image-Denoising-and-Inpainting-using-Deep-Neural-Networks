import torch
from torch import optim
from model import DenoisingAutoencoder, SSDA
from utils import prepare_data
import os
import matplotlib.pyplot as plt
import time  # added for timing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pre_train_loss = []
fine_tune_loss = []


def pretrain_da(model, dataloader, epochs=15, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out, h = model(xb)
            loss = criterion(out, yb) + model.sparsity_loss(h) + model.weight_regularization()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pre_train_loss.append(total_loss)
        epoch_end = time.time()
        elapsed = epoch_end - start_time
        epoch_time = epoch_end - epoch_start
        est_total_time = (elapsed / (epoch + 1)) * epochs
        est_remaining = est_total_time - elapsed
        print(f"[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s, Remaining: {est_remaining/60:.2f}min")

    total_time = time.time() - start_time
    print(f"Pretraining completed in {total_time/60:.2f} minutes.\n")


def fine_tune(model, dataloader, epochs=25, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb) + model.weight_regularization()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        fine_tune_loss.append(total_loss)
        epoch_end = time.time()
        elapsed = epoch_end - start_time
        epoch_time = epoch_end - epoch_start
        est_total_time = (elapsed / (epoch + 1)) * epochs
        est_remaining = est_total_time - elapsed
        print(f"[Fine-tune] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s, Remaining: {est_remaining/60:.2f}min")

    total_time = time.time() - start_time
    print(f"Fine-tuning completed in {total_time/60:.2f} minutes.\n")


if __name__ == "__main__":
    # sigma = 50
    patch_size = 16
    dataloader = prepare_data(patch_size= patch_size)
    input_dim = 16 * 16
    hidden_dims = [512, 256]

    dae1 = DenoisingAutoencoder(input_dim, hidden_dims[0]).to(device)
    print("Pretraining DAE1...")
    pretrain_da(dae1, dataloader)

    h1_x, h1_y = [], []
    dae1.eval()
    for xb, yb in dataloader:
        with torch.no_grad():
            xb, yb = xb.to(device), yb.to(device)
            _, h_x = dae1(xb)
            _, h_y = dae1(yb)
            h1_x.append(h_x.cpu())
            h1_y.append(h_y.cpu())
    h1_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.cat(h1_x), torch.cat(h1_y)),
        batch_size=128, shuffle=True
    )

    dae2 = DenoisingAutoencoder(hidden_dims[0], hidden_dims[1]).to(device)
    print("Pretraining DAE2...")
    pretrain_da(dae2, h1_loader)

    ssda = SSDA(input_dim, hidden_dims).to(device)
    ssda.load_state_dict({
        **{f'dae1.{k}': v for k, v in dae1.state_dict().items()},
        **{f'dae2.{k}': v for k, v in dae2.state_dict().items()}
    }, strict=False)

    os.makedirs("model", exist_ok=True)
    print("Fine-tuning SSDA...")
    fine_tune(ssda, dataloader)
    torch.save(ssda.state_dict(), "model/ssda_inpainting.pth")
    print("Model saved to model/ssda_inpainting.pth")

    # Save the training losses as graphs in subplots
    os.makedirs("result", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pre_train_loss, label='Pretrain Loss')
    plt.title('Pretraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fine_tune_loss, label='Fine-tune Loss')
    plt.title('Fine-tuning Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("result/loss_graphs.png")
    plt.show()
