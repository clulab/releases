from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def plot_tsne(embeddings, labels, filename, plot_only=500, colors=None, init='pca'):
    tsne = TSNE(perplexity=30, n_components=2, init=init, n_iter=5000)
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    labels = labels[:plot_only]
    if colors is not None:
        colors = colors[:plot_only]
    plot_with_labels(low_dim_embs, labels, filename, colors)
    return low_dim_embs

def do_tsne(embedding, initialization='pca', perplex=30,lr=2e2):
    tsne = TSNE(method='exact',perplexity=perplex, n_components=2, init=initialization, n_iter=1000, learning_rate=lr, verbose=2)
    return tsne.fit_transform(embedding)

def plot_with_labels(low_dim_embs, labels, filename, colors=None):
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        if colors is not None:
            plt.scatter(x, y, c=colors[i])
        else:
            plt.scatter(x, y)
        # plt.annotate(label,
        #         xy=(x, y),
        #         xytext=(5, 2),
        #         textcoords='offset points',
        #         ha='right',
        #         va='bottom')
    plt.savefig(filename)
    plt.close()
