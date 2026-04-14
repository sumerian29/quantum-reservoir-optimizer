import matplotlib.pyplot as plt
import numpy as np

def plot_well_selection(x, production):
    plt.figure()
    plt.bar(range(len(x)), production, color='gray', label='All wells')
    
    selected = np.where(x == 1)[0]
    plt.bar(selected, production[selected], color='green', label='Selected wells')
    
    plt.title("Well Selection Optimization")
    plt.xlabel("Well Index")
    plt.ylabel("Production Potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)
    plt.show()


def plot_production_vs_selection(x, production):
    plt.figure()
    plt.scatter(range(len(production)), production, c=x, cmap='coolwarm')
    plt.title("Production vs Selection Decision")
    plt.xlabel("Well Index")
    plt.ylabel("Production")
    plt.colorbar(label="Selected (1) / Not Selected (0)")
    plt.tight_layout()
    plt.savefig("production_scatter.png", dpi=300)
    plt.show()
