import streamlit as st
import regex as re
import fitz
# import pke
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util

checkpoint = "sadickam/sdg-classification-bert"
model = SentenceTransformer("all-MiniLM-L6-v2")
MIN_WORD_CNT = 10
CNT = 100
reg_str = r'[^!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\0-9a-zA-Z]'
classifier = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)
df_sdg = pd.read_excel("./data/SDGs/sdg.xlsx")


def prep_text(text):
    """
    function for preprocessing text
    """

    clean_sents = []
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [
            str(word_token).strip().lower() for word_token in sent_token.split()
        ]
        clean_sents.append(" ".join((word_tokens)))
    joined = " ".join(clean_sents).strip(" ")
    joined = re.sub(r"`", "", joined)
    joined = re.sub(r'"', "", joined)
    return joined


@st.cache_resource()
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint)


@st.cache_resource()
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


def get_cnt(text):
    cnt = 0
    for word in text.split():
        if word.isalpha():
            cnt += 1
    return cnt


def get_text(block_lst):

    text_lst = []
    for block in block_lst:
        if block[6] != 0:
            continue

        text = block[4]
        text = text.replace("fi ", "fi")

        if get_cnt(text) < MIN_WORD_CNT:
            continue

        text_lst.append(text.replace("-\n", ""))

    return "\n".join(text_lst)


def get_sentences(fname, skip_page=(0,)):

    doc = fitz.open(fname)

    sent_lst = []
    for page_no, page in enumerate(doc):

        if page_no + 1 in skip_page:
            continue

        block_lst = page.get_text("blocks")
        text = get_text(block_lst)

        for i, sentence in enumerate(sent_tokenize(text)):
            r_sent = " ".join(sentence.split())
            sent_lst.append(r_sent)

    doc.close()

    return sent_lst


# def load_sdg_embeddings():
#     df_sdg["sentence"] = df_sdg["sentence"].str.replace(reg_str, " ", regex=True)
#     sdg_sentences = df_sdg["sentence"].tolist()
#     sdg_embeddings = model.encode(sdg_sentences, convert_to_tensor=True)
#     return sdg_embeddings


# def calculate_cosine_scores(sdg_embeddings, sentences):
#     sentences = [re.sub(reg_str, " ", sentence) for sentence in sentences]
#     embedding2 = model.encode(sentences, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(sdg_embeddings, embedding2)
#     # make an array of the scores for each goal hence having cosine scores for each 17 goals
#     cosine_scores = cosine_scores.numpy()

#     return cosine_scores


def sentiment_analysis(sent_list):
    result = []
    CNT = 100
    LEN = len(sent_list)

    for i in range(0, LEN, CNT):
        off_e = (i + CNT) if (i + CNT) < LEN else LEN
        # Max length of sentence: 512
        res = classifier(sent_list[i:off_e], truncation=True)
        result.extend(res)

    return result


def carbon_assessment(hardware_type, hours_used, provider, region):
    gpus_df = pd.read_csv("./data/carbon/gpus.csv")
    impact_df = pd.read_csv("./data/carbon/impact.csv")

    power_consumption = gpus_df[gpus_df["name"] == hardware_type]["tdp_watts"].iloc[0]
    power_consumption = power_consumption / 1000
    carbon_produced_per_kwh = impact_df[
        (impact_df["providerName"] == provider) & (impact_df["region"] == region)
    ]["impact"].iloc[0]
    offset_ratio = impact_df[
        (impact_df["providerName"] == provider) & (impact_df["region"] == region)
    ]["offsetRatio"].iloc[0]
    country = impact_df[
        (impact_df["providerName"] == provider) & (impact_df["region"] == region)
    ]["country"].iloc[0]
    state = impact_df[
        (impact_df["providerName"] == provider) & (impact_df["region"] == region)
    ]["state"].iloc[0]
    city = impact_df[
        (impact_df["providerName"] == provider) & (impact_df["region"] == region)
    ]["city"].iloc[0]
    carbon_produced_per_kwh = carbon_produced_per_kwh / 1000
    carbon_emission = power_consumption * hours_used * carbon_produced_per_kwh
    ice_kms_driven = (carbon_emission * 1.609344) / 0.398
    kgs_coal_burnt = carbon_emission / (0.905 * 2.204623)
    sequestered_trees = carbon_emission / 60
    min_impact = impact_df["impact"].min()
    min_impact_provider = impact_df[impact_df["impact"] == min_impact][
        "providerName"
    ].iloc[0]
    min_impact_region = impact_df[impact_df["impact"] == min_impact]["region"].iloc[0]
    min_carbon_emission = power_consumption * hours_used * (min_impact / 1000)

    return (
        power_consumption,
        carbon_produced_per_kwh,
        offset_ratio,
        country,
        state,
        city,
        carbon_emission,
        ice_kms_driven,
        kgs_coal_burnt,
        sequestered_trees,
        min_impact_provider,
        min_impact_region,
        min_carbon_emission,
    )


# def extract_keywords(text):
#     pos = {"NOUN", "PROPN", "ADJ"}
#     extractor = pke.unsupervised.SingleRank()
#     # 2. load the content of the document.
#     extractor.load_document(input=text, language="en", normalization=None)

#     # 3. select the longest sequences of nouns and adjectives as candidates.
#     extractor.candidate_selection(pos=pos)

#     # 4. weight the candidates using the sum of their word's scores that are
#     #    computed using random walk. In the graph, nodes are words of
#     #    certain part-of-speech (nouns and adjectives) that are connected if
#     #    they occur in a window of 10 words.
#     extractor.candidate_weighting(window=10, pos=pos)

#     # 5. get the 10-highest scored candidates as keyphrases
#     keyphrases = extractor.get_n_best(n=10)
#     # convert to list
#     keyphrases = [keyphrase[0] for keyphrase in keyphrases]

#     return keyphrases
