import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")

# Initialize the OpenAI API
openai = OpenAI(openai_api_key=openai_api_key)

# Define the LangChain prompt template
template = """
You are an AI assistant helping find the sentiment of news articles.
Given the following news article, analyze its sentiment (positive, negative, or neutral):

Article: {article}
"""

prompt = PromptTemplate(template=template, input_variables=["article"])
llm_chain = LLMChain(prompt=prompt, llm=openai)

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=newsapi_key)

def get_news_article(query):
    articles = newsapi.get_everything(q=query, sort_by='relevancy')
    return articles['articles']

def extract_article_content(articles):
    max_tokens = 20
    return [article['content'][:max_tokens] for article in articles if article['content']]

def analyzes_sentiment(query):
    articles = get_news_article(query)
    content = extract_article_content(articles)

    if content:
        article_text = content[0]
        result = llm_chain.run({"article": article_text})
        return result
    else:
        return "No articles found for the given query! Try another one."

# Example usage
# query = "artificial intelligence"
# print(analyze_sentiment(query))
