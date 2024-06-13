import numpy as np
from sklearn.pipeline import Pipeline
import sys
sys.path.append('/Users/luis.caporal/Documents/Notebooks/DS_TEST/DS_Project')
from scripts.extraction.extract_data import load_xml_files
from scripts.transformers.pipelines import RemoveLastParagraphTransformer, RemoveHTMLTagsTransformer, TextPreprocessingSwitcher, \
    LDAModelTransformer, TopicLabelingTransformer
from scripts.utils.functions import plot_wordclouds_for_topics



def main(bert=False, topics=10, iterations=25, wordclouds=True):

    data, files, _, _ = load_xml_files('2020')

    preprocessing_pipeline = Pipeline([
        ('remove_last_paragraph', RemoveLastParagraphTransformer()),
        ('remove_html_tags', RemoveHTMLTagsTransformer()),
        ('text_preprocessing', TextPreprocessingSwitcher(use_bert=bert))
    ])


    lda_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('lda_model', LDAModelTransformer(num_topics=topics, iterations=iterations)),
        ('topic_labeling', TopicLabelingTransformer())
    ])


    """ LDA model fitting """
    lda_pipeline.fit(data)
    lda_model = lda_pipeline.named_steps['lda_model']
    topic_labels = lda_pipeline.named_steps['topic_labeling'].transform(X=None)
    document_topic_distribution = [lda_model.model.get_document_topics(doc) for doc in lda_model.corpus]

    if wordclouds:

        plot_wordclouds_for_topics(topics, lda_model.model.show_topics(formatted=False), topic_labels)
        

    return lda_model, document_topic_distribution, files, topic_labels

if __name__ == '__main__':
    main()