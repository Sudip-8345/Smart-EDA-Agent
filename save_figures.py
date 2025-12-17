import tempfile
import matplotlib.pyplot as plt

def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight", dpi= 150)
    plt.close(fig)
    return f.name