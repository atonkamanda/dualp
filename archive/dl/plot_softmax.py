import matplotlib.pyplot as plt


"""def compare_beliefs(softmax1, softmax2,kl,name1='Softmax1',name2="Softmax2"):
    categories = range(len(softmax1))
    colors = ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'brown', 'orange']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for i, softmax in enumerate([softmax1, softmax2]):
        ax = [ax1, ax2][i]
        for j, b in enumerate(softmax):
            ax.bar(j, b, color=colors[j], width=1)
        ax.set_xlim(-1, len(softmax))
        ax.set_xticks(categories)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Probability')
        ax.set_title([name1, name2][i])
    fig.legend(['KL Divergence: {:.2f}'.format(kl)], loc='upper center')
    plt.tight_layout()
    plt.show()"""

def compare_beliefs(softmax1, softmax2, kl, name1='Softmax1', name2="Softmax2"):
    categories = range(len(softmax1))
    colors = ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'brown', 'orange']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 2], 'wspace': 0.3})
    for i, softmax in enumerate([softmax1, softmax2]):
        ax = [ax1, ax2][i]
        for j, b in enumerate(softmax):
            ax.bar(j, b, color=colors[j], width=0.8, edgecolor='black')
        ax.set_xlim(-1, len(softmax))
        ax.set_xticks(categories)
        ax.set_xlabel('Categories', fontsize=12)
        if i == 0:
            ax.set_ylabel('Probability', fontsize=12)
        ax.set_title([name1, name2][i], fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
    fig.text(0.5, 0.9, 'KL: {:.2f}'.format(kl), ha='center', fontsize=12, fontweight='bold')
    #plt.tight_layout()
    plt.show()
import numpy as np
belief1 = [0.1, 0.2, 0.7]
belief2 = [0.2, 0.3, 0.5]
# To numpy array
belief1 = np.array(belief1)
belief2 = np.array(belief2)
compare_beliefs(belief1, belief2,2)
