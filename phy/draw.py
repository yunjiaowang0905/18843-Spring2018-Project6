import seaborn as sns
import matplotlib.pyplot as plt

def draw_heatmap(data, i_t):
    heatmap = sns.heatmap(data, cmap='Reds')
    filename = "./newdata/" + str(i_t) + "_iter_output.png"
    heatmap.figure.savefig(filename)
    plt.clf()