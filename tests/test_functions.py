import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/luis.caporal/Documents/Notebooks/DS_TEST/DS_Project')
from scripts.processing.lda_model import main


def test_get_model_coherence():

    """ Test to calculate model coherence based on different parameters """
    on = False

    if not on:
        assert 1 == 1
    else:

        X = [[], [], []]
        X2 = [[], [], []]
        Y1 = [[], [], []]
        Y2 = [[], [], []]
        Y3 = [[], [], []]
        Y4 = [[], [], []]

        for i in range(0,3):
            for j in range(1, 26, 1):
                X[i].append(j)
                print('iteración', i, j)
                simple_model, _, _, _  = main(bert=False, topics=j, iterations=30)
                bert_model, _, _, _ = main(bert=True, topics=j, iterations=30)
                coherence = simple_model.coherence
                b_coherence = bert_model.coherence
                Y1[i].append(coherence)
                Y2[i].append(b_coherence)
                print('iteración', i, j)

        simple_topics = pd.DataFrame({'X': np.array(X[0]).T, 'Y_1': np.array(Y1[0]).T, 
                        'Y_2': np.array(Y1[1]).T, 'Y_3': np.array(Y1[0]).T})

        bert_topics = pd.DataFrame({'X': np.array(X[0]).T, 'Y_1': np.array(Y2[0]).T, 
                        'Y_2': np.array(Y2[1]).T, 'Y_3': np.array(Y2[0]).T})
        
        mean1 = bert_topics.groupby('X').mean()
        std1 = bert_topics.groupby('X').std()
        mean2 = simple_topics.groupby('X').mean()
        std2 = simple_topics.groupby('X').std()
        mean1['mean_Y'] = mean1.mean(axis=1)
        std1['std_Y'] = std1.mean(axis=1)
        mean2['mean_Y'] = mean2.mean(axis=1)
        std2['std_Y'] = std2.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.errorbar(mean1.index, mean1['mean_Y'], yerr=std1['std_Y'], fmt='-o', label='Bert token')
        plt.errorbar(mean2.index, mean2['mean_Y'], yerr=std2['std_Y'], fmt='-s', label='Simple token')

        plt.xlabel('Num. topics')
        plt.ylabel('Model Coherence')
        plt.title('Average coherence of the model as a function of number of topics')
        plt.legend()
        plt.grid(True)
        plt.show()
        

        assert len(simple_topics) == 25
        assert len(bert_topics) == 25