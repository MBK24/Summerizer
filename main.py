import langchain_helper as lch 
import streamlit as st 
import textwrap


st.title("Summerizer")

with st.sidebar:
    with st.form(key = 'my_form'):
        youtubeUrl = st.sidebar.text_area("What is the Youtube URL?", max_chars = 50)

        question = st.sidebar.text_area(label = "What would you like to know ?", max_chars = 50, key = "query")

        submitButton = st.form_submit_button(label = " Submit")

if question and youtubeUrl:
    db = lch.youtubeVectorDb(youtubeUrl)
    response, docs =  lch.responseFromQuery(db, query = question)
    st.subheader("Answer:")
    response = response.replace("\n"," ")
    st.text(textwrap.fill(response, width = 80))