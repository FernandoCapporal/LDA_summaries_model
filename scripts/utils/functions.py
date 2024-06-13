import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


def sturges_rule(col):

    """ Function to determine the correct number of bins in a hsitogram """
    n = len(col)
    k = 1 + np.log2(n)
    return int(k)


def gratitude_counter(data):

    """ Function to verify howe many files end with awards to the NFS"""
    cnt = 0
    for i in range(0, len(data)):
        if 'This award reflects' in data[i]:
            cnt+=1
    return cnt


def remove_html_tags(text):

    """ Function to remove tags like br and lt """
    text_without_html = BeautifulSoup(text, "html.parser").get_text()
    text_without_html = re.sub(r'\s+', ' ', text_without_html).strip()
    br_pattern = re.compile(r'<br\s*/?>\s*<br\s*/?>', re.IGNORECASE)
    return br_pattern.sub('', text_without_html)


def remove_last_paragraph(text):

    """ Function to remove the lasth paragraph (awards) """
    paragraphs = text.split('This award reflects')
    if len(paragraphs) >= 2:
        new_text = '. '.join(paragraphs[:-1])
    else:
        new_text = paragraphs[0]
    return new_text



def plot_wordclouds_for_topics(num_topics, topics, legeds):

    """ Function to pretty plot wordclouds based on number of topics """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    num_rows = (num_topics // 3) + (1 if num_topics % 3 != 0 else 0)
    
    fig = plt.figure(figsize=(10, num_rows * 5))
    gs = gridspec.GridSpec(num_rows, 6, figure=fig, wspace=0.0, hspace=0.0)

    for i in range(num_topics):
        row = i // 3
        if row == num_rows - 1:
            width_of_bottom_plots = 2 * (num_topics - (3 * (num_rows - 1)))
            start_col = (6 - width_of_bottom_plots) // 2 + ((i % 3) * 2)
            end_col = start_col + 2
        else:
            start_col = (i % 3) * 2
            end_col = start_col + 2
        
        ax = fig.add_subplot(gs[row, start_col:end_col])
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        ax.imshow(cloud)
        ax.set_title(f'{legeds[i]}', fontdict=dict(size=20, color='k'))
        ax.axis('off')

    plt.tight_layout(pad=0.0) 
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
