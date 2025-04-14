import streamlit as st
from newslang import llm_chain
from newslang import analyzes_sentiment


st.title("News Search and Sentiment Analysis Tool")

st.write("Enter a query")

query = st.text_input("Query")

if st.button("Analyse NEWS"):
    if query:
        articles = analyzes_sentiment(query)
        if articles:
            article_content = articles[0]
            response = llm_chain.run({'article': article_content})

            st.write(response)
        else:
            st.write("No relevant articles found")

    else:
        st.write("please enter a query")
