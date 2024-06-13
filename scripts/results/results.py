import sys
sys.path.append('/Users/luis.caporal/Documents/Notebooks/DS_TEST/DS_Project')
from scripts.processing.lda_model import main
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import LegendItem, Legend
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tsne_lda(document_topic_distribution, num_topics, topic_labels):

    topic_weights = []
    for i, row_list in enumerate(document_topic_distribution):
        topic_weights.append([w for i, w in row_list])

    arr = pd.DataFrame(topic_weights).fillna(0).values
    arr = arr[np.amax(arr, axis=1) > 0.35]
    topic_num = np.argmax(arr, axis=1)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    output_file("lda_tsne_clusters.html")
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(num_topics), 
                  width=900, height=700)
    
    for i in range(num_topics):
        plot.scatter(x=tsne_lda[topic_num == i, 0], y=tsne_lda[topic_num == i, 1], 
                     color=mycolors[i], legend_label=topic_labels[i])

    plot.xaxis.axis_label = 't-SNE Dimension 1'
    plot.yaxis.axis_label = 't-SNE Dimension 2'

    legend_items = []
    for i in range(num_topics):
        legend_items.append(LegendItem(label=topic_labels[i], renderers=[plot.scatter([], [], color=mycolors[i])], index=i))

    plot.add_layout(Legend(items=legend_items, location='top_left'), 'right')

    show(plot)


def plot_count_topics(count):

    words = [item[2] for item in count]
    word_counts = Counter(words)

    words = list(word_counts.keys())
    counts = list(word_counts.values())

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette='deep', alpha=0.8)
    plt.xlabel('Number of documents')
    plt.ylabel('Topic')
    plt.title('Topic count')
    plt.show()


def get_results(wordclouds=False, show_distribution=False, show_counter=False):

    num_topics = 6
    lda_model, document_topic_distribution, files, topic_labels = main(bert=True, topics=num_topics, wordclouds=wordclouds)

    if num_topics <= 10 and show_distribution:
        plot_tsne_lda(document_topic_distribution, num_topics, topic_labels)

    dominant_topics = []
    classifier = {}

    for idx in range(0, len(document_topic_distribution)):
        dominant_topic = max(document_topic_distribution[idx], key=lambda x: x[1])[0]
        classifier[files[idx]] = {'Topic': topic_labels[dominant_topic], 'Summary': lda_model.original_summaries[idx]}
        dominant_topics.append((files[idx], dominant_topic, topic_labels[dominant_topic]))

    if show_counter:
        plot_count_topics(dominant_topics)

    return classifier

if __name__ == '__main__':
    get_results()