import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
import leafmap.foliumap as leafmap

from utils import (
    prep_text,
    load_model,
    load_tokenizer,
    get_sentences,
    sentiment_analysis,
    carbon_assessment,
    # extract_keywords,
)

import plotly.express as px
import pandas as pd
import folium
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np

from wordcloud import WordCloud
import torch
import nltk

nltk.download("punkt")

from codecarbon import track_emissions


@track_emissions(save_to_api=True)
def main():
    # your code

    st.set_page_config(
        page_title="Climate Cognizance",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon="🚦",
    )

    st.title("Climate Cognizance")

    with st.sidebar:
        selected = option_menu(
            None,
            ["Home", "Sustainability Goals", "Carbon Assessment"],
            icons=["house", "check", "recycle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        st.markdown("### Welcome to the Climate and Sustainability App")
        st.text(
            "This app is designed to help you assess and benchmark sustainability and climate impact of your business or project."
        )
        st.text("Use the sidebar to navigate to the various sections of the app")

    elif selected == "Sustainability Goals":

        label_list = [
            "GOAL 1: No Poverty",
            "GOAL 2: Zero Hunger",
            "GOAL 3: Good Health and Well-being",
            "GOAL 4: Quality Education",
            "GOAL 5: Gender Equality",
            "GOAL 6: Clean Water and Sanitation",
            "GOAL 7: Affordable and Clean Energy",
            "GOAL 8: Decent Work and Economic Growth",
            "GOAL 9: Industry, Innovation and Infrastructure",
            "GOAL 10: Reduced Inequality",
            "GOAL 11: Sustainable Cities and Communities",
            "GOAL 12: Responsible Consumption and Production",
            "GOAL 13: Climate Action",
            "GOAL 14: Life Below Water",
            "GOAL 15: Life on Land",
            "GOAL 16: Peace, Justice and Strong Institutions",
        ]

        st.header("Sustainability Goals")

        with st.expander("About this app", expanded=False):
            st.write(
                """
                - Artificial Intelligence (AI) tool for automatic classification of text with respect to the UN Sustainable Development Goals (SDG)
                - Note that 16 out of the 17 SDGs are covered
                - This tool is for sustainability assessment and benchmarking and is not limited to a specific industry
                - The model powering this app was developed using the OSDG Community Dataset (OSDG-CD) [Link - https://zenodo.org/record/5550238#.Y8Sd5f5ByF5]
                """
            )

        tokenizer_ = load_tokenizer()
        model = load_model()

        st.markdown("##### Upload PDF file or Enter Text")
        with st.form(key="my_form"):
            c1, c2 = st.columns([1, 1])
            with c1:
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            with c2:
                Text_entry = st.text_area("Paste or type text in the box below")

            submitted = st.form_submit_button(label="Predict SDG!")

        if submitted:

            if uploaded_file is not None:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_file = uploaded_file.name

                sent_list = get_sentences(uploaded_file)
                sent_list.append(Text_entry)

                sent_str = " ".join(sent_list)
                text = sent_str + Text_entry
            else:
                text = Text_entry

            if text == "":
                st.warning(
                    """This app needs text input to generate predictions. Kindly type text into 
                    the above **"Text Input"** box" or upload a PDF file""",
                    icon="⚠️",
                )

            elif text != "":
                joined_clean_sents = prep_text(text)

                tokenized_text = tokenizer_(
                    joined_clean_sents,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                text_logits = model(**tokenized_text).logits
                predictions = torch.softmax(text_logits, dim=1).tolist()[0]
                predictions = [round(a, 3) for a in predictions]

                pred_dict = dict(zip(label_list, predictions))

                sorted_preds = sorted(
                    pred_dict.items(), key=lambda x: x[1], reverse=True
                )

                u, v = zip(*sorted_preds)
                x = list(u)
                y = list(v)
                df2 = pd.DataFrame()
                df2["SDG"] = x
                df2["Likelihood"] = y

                c1, c2, c3 = st.columns([1.5, 0.5, 1])

                with c1:
                    st.markdown("##### Prediction outcome")
                    fig = px.bar(df2, x="Likelihood", y="SDG", orientation="h")

                    fig.update_layout(
                        barmode="stack",
                        template="seaborn",
                        font=dict(family="Arial", size=14, color="white"),
                        autosize=False,
                        width=800,
                        height=500,
                        xaxis_title="Likelihood of SDG",
                        yaxis_title="Sustainable development goals (SDG)",
                        legend_title="Topics",
                    )

                    fig.update_xaxes(
                        tickangle=0,
                        tickfont=dict(family="Arial", color="white", size=14),
                    )
                    fig.update_yaxes(
                        tickangle=0,
                        tickfont=dict(family="Arial", color="white", size=14),
                    )
                    fig.update_annotations(font_size=14)

                    st.plotly_chart(fig, use_container_width=False)
                    st.success("SDG successfully predicted. ", icon="✅")

                with c3:
                    st.header("")
                    predicted = st.markdown(
                        "###### Predicted " + str(sorted_preds[0][0])
                    )
                    Prediction_confidence = st.metric(
                        "Prediction confidence",
                        (str(round(sorted_preds[0][1] * 100, 1)) + "%"),
                    )
                # keywords = extract_keywords(text)
                # keywords = [word for phrase in keywords for word in phrase.split()]
                # keywords = list(set(keywords))
                # st.write("Keywords extracted from the text: ", keywords)

                # wordcloud = WordCloud(
                #     width=800,
                #     height=800,
                #     background_color="white",
                #     stopwords=None,
                #     min_font_size=10,
                # ).generate(" ".join(keywords))
                # plt.figure(figsize=(8, 8), facecolor=None)
                # plt.imshow(wordcloud)
                # plt.axis("off")
                # plt.tight_layout(pad=0)
                # plt.show()
                # st.pyplot(plt)

                # cosine_scores = calculate_cosine_scores(sdg_embeddings, sent_list)
                sentiment_scores = sentiment_analysis(sent_list)
                df_neg = pd.DataFrame(sentiment_scores)
                df_neg["sentence"] = sent_list[: len(sent_list)]
                df_neg = df_neg[df_neg["label"] == "NEGATIVE"].nlargest(10, "score")

                # display the top 10 negative sentences in a data editor table
                st.write("Top 10 negative sentences")
                edited_df = st.data_editor(
                    df_neg,
                    column_config={
                        "score": st.column_config.NumberColumn(
                            "Sentiment Score",
                            help="Sentiment score of the sentence (0-1)",
                            min_value=0,
                            max_value=1,
                        ),
                        "sentence": st.column_config.TextColumn(
                            "Sentence", help="The text of the sentence"
                        ),
                    },
                    hide_index=True,
                )

    elif selected == "Carbon Assessment":

        gpus_df = pd.read_csv("./data/carbon/gpus.csv")
        impact_df = pd.read_csv("./data/carbon/impact.csv")

        st.markdown("##### Carbon Emission Assessment")
        with st.form(key="carbon_form"):
            hardware_type = st.selectbox(
                "Select hardware type", gpus_df["name"].unique()
            )
            hours_used = st.number_input(
                "Enter hours used", min_value=0, value=0, step=1
            )
            provider = st.selectbox(
                "Select cloud provider", impact_df["providerName"].unique()
            )
            region = st.selectbox("Select region", impact_df["region"].unique())
            submitted = st.form_submit_button(label="Calculate Carbon Emission")

        if submitted:
            (
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
            ) = carbon_assessment(hardware_type, hours_used, provider, region)

            st.write("CARBON EMITTED")
            st.write(round(carbon_emission, 2))
            st.write(
                f"Power consumption x Time x Carbon Produced Based on the Local Power Grid:\n{power_consumption * 1000 }W x {hours_used}h = {power_consumption*hours_used} kWh x {carbon_produced_per_kwh} kg eq. CO2/kWh = {carbon_emission} kg eq. CO2"
            )
            st.write(
                f"Had this model been run in {min_impact_provider}'s {min_impact_region} region, the carbon emitted would have been of {min_carbon_emission} kg eq. CO2"
            )
            st.write(f"{round(carbon_emission, 2)} kg of CO2eq. is equivalent to:")
            st.write(round(ice_kms_driven, 2), "Km driven by an average ICE car [1]")
            st.write(round(kgs_coal_burnt, 2), "Kgs of coal burned [2]")
            st.write(
                round(sequestered_trees, 2),
                "Tree seedlings sequestering carbon for 10 years [3]",
            )

        site_df = pd.read_csv("./data/carbon/CCS Map Data Jan2023.csv")
        
        st.markdown("## Carbon Sequestration sites")

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=37.76,
                    longitude=-122.4,
                    zoom=3,
                    pitch=40,
                ),
                layers=[
                    pdk.Layer(
                        "HexagonLayer",
                        data=site_df,
                        get_position="[Longitude, Latitude]",
                        radius=20000,
                        elevation_scale=40,
                        elevation_range=[800, 1000],
                        pickable=True,
                        extruded=True,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=site_df,
                        get_position="[Longitude, Latitude]",
                        # green color
                        get_fill_color=[0, 255, 0],
                        get_color=[
                            [0, 25, 0, 25],
                            [0, 85, 0, 85],
                            [0, 127, 0, 127],
                            [0, 170, 0, 170],
                            [0, 190, 0, 190],
                            [0, 255, 0, 255],
                        ],
                        get_radius=20000,
                    ),
                ],
            )
        )

        folium_map = folium.Map(zoom_start=8)

        for index, row in site_df.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=row["Project Name"],
                icon=folium.Icon(color="green", icon="info-sign"),
            ).add_to(folium_map)

       
        folium_static(folium_map)


if __name__ == "__main__":
    main()
