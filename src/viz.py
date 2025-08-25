# viz_utils.py
import matplotlib.pyplot as plt

def plot_losses(losses, out_path="src/artifacts/training_loss.png", logy=False):
    """
    Plot loss curve and save to file.

    Args:
        losses (list of float): loss values (per step or per log interval)
        out_path (str): where to save the plot
        logy (bool): plot y-axis in log scale
    """
    plt.figure(figsize=(8,5))
    plt.plot(losses, marker=".", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss (Skip-gram NS)")
    if logy:
        plt.yscale("log")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {out_path}")