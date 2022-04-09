import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy_streamlit import visualize_ner
import spacy_streamlit
from st_aggrid import AgGrid
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline
from statistics import mean


def load_sentiment_model():
    model = DistilBertForSequenceClassification.from_pretrained('final')
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased', num_labels=3)
    sentiment = pipeline("sentiment-analysis",
                         model=model, tokenizer=tokenizer)
    return sentiment


def get_score(text):
    label = sentiment(text)[0]['label']
    return 'Negative' if '0' in label else ('Postive' if '2' in label else 'Neutral')


def convert_score_word(score):
    return 'Negative' if score < 0 else 'Positive' if score > 0 else 'Neutral'


def get_org_score(text):
    org_list = {}
    sentences = sentencizer(text)
    for sent in sentences.sents:
        doc = nlp(sent.text)
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                label = sentiment(text)[0]['label']
                int_label = -1 if '0' in label else (1 if '2' in label else 0)
                if ent.text in org_list:
                    org_list[ent.text].append(int_label)
                else:
                    org_list[ent.text] = [int_label]
    # return pd.DataFrame.from_dict({org: (mean(scores)-1) for org, scores in org_list.items()})
    return [{'org_name': org, 'score': convert_score_word(mean(scores))} for org, scores in org_list.items()]


nlp = spacy.load('model-best')
sentiment = load_sentiment_model()
sentencizer = spacy.load("en_core_web_sm")
# default_text = "Samsung has made 200% profit, earning $50M this year."
default_text = st.text_area("Message", height=100)

models = ["model-last"]
doc = nlp(default_text)
visualizers = ["ner"]
# visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
score = get_score(default_text)
df_score = get_org_score(default_text)
# print(df_score)
df = pd.read_json('all_data_with_cat.json')
# st.write(df[['title','summary']])


st.title('Organization Sentiment Analysis')
st.subheader('Dataset')
AgGrid(df[['title', 'text']], theme='streamlit',
       editable=True, fit_columns_on_grid_load=True)
st.subheader('Overall Sentiment Score')
st.text(score)
st.subheader('Detailed Sentiment Score')
st.dataframe(df_score)
# spacy_streamlit.visualize(models, default_text, visualizers=visualizers)
visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
