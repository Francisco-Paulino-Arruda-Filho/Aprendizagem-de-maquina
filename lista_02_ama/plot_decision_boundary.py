import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, clf=None, steps=300, colors=None):
    # Definir cores padrão se não passar
    if colors is None:
        colors = ['green', 'blue', 'red', 'purple', 'orange']
    
    # Criar grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, steps),
        np.linspace(y_min, y_max, steps)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotar regiões de decisão (se tiver classificador)
    if clf is not None:
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf(grid)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, levels=len(np.unique(y))-1)

    # Plotar pontos
    labels = np.unique(y)
    for label in labels:
        ax.scatter(
            X[y == label, 0],
            X[y == label, 1],
            s=120,
            marker='o',
            color=colors[int(label)],
            label=f'Classe {label}'
        )

    ax.legend()
    ax.set_title("Decision Boundary")
    plt.show()