import subprocess
import streamlit as st

args = [
    "apt-get install sudo",
    "sudo apt-get update",
    "sudo apt-get install poppler-utils",
    "sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn"
    "pip install unstructured-pytesseract",
    "pip install tesseract-ocr"
]
st.info(f"Running '{' '.join(args)}'")
result = subprocess.run(args, capture_output=True, text=True, shell=True, check=True)

import io
import re
import os
import uuid
import utils
import base64
import openai
from PIL import Image
from typing import Any
from pydantic import BaseModel
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf

fpath = "Data/Extracted Data"

# Title
st.title("Multimodal RAG PDF Q&A using GPT-4o")
st.write("Upload Pdf's and chat with their content")

## Sidebar for settings
st.sidebar.title("Settings")
apikey = st.sidebar.text_input("Enter your Open AI API Key:", type="password")

## Upload PDF
uploaded_file=st.file_uploader("Choose A Pdf file", type="pdf", accept_multiple_files=False)

## Process uploaded  PDF's
if uploaded_file and apikey:
    raw_pdf_elements=partition_pdf(
        filename=uploaded_file,                  # mandatory
        strategy="hi_res",                                 # mandatory to use ``hi_res`` strategy
        extract_images_in_pdf=True,                       # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],          # optional
        extract_image_block_to_payload=False,                  # optional
        extract_image_block_output_dir=fpath,  # optional - only works when ``extract_image_block_to_payload=False``
    )

    text=[]
    img=[]
    table=[]

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Header" in str(type(element)):
                    text.append(str(element))
        elif "unstructured.documents.elements.Image" in str(type(element)):
                    img.append(str(element))
        elif "unstructured.documents.elements.Table" in str(type(element)):
                    table.append(str(element))
        elif "unstructured.documents.elements.Footer" in str(type(element)):
                    text.append(str(element))
        elif "unstructured.documents.elements.Title" in str(type(element)):
                    text.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
                    text.append(str(element))
        elif "unstructured.documents.elements.Text" in str(type(element)):
                    text.append(str(element))
        elif "unstructured.documents.elements.ListItem" in str(type(element)):
                    text.append(str(element))
    
    # Table Summaries
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
        These summaries will be embedded and used to retrieve the raw table elements. \
        Give a concise summary of the table that is well optimized for retrieval. Table:{element} """
    prompt_table = ChatPromptTemplate.from_template(prompt_text)
    table_summaries = utils.text_table_summarizer(table, apikey, prompt_table)

    # Text Summaries
    prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text elements. \
        Give a concise summary of the table or text that is well optimized for retrieval.text: {element} """
    prompt_for_text = ChatPromptTemplate.from_template(prompt_text)
    text_summaries = utils.text_table_summarizer(text, apikey, prompt_for_text)

    # Image Summaries
    img_base64_list, image_summaries = utils.generate_img_summaries(fpath)

    # Mutlivector Retreiver
    vectorstore = Chroma(
        collection_name="mm_rag", embedding_function=OpenAIEmbeddings()
    )
    retriever_multi_vector_img = utils.create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        text,
        table_summaries,
        table,
        image_summaries,
        img_base64_list,
    )

    # RAG chain
    chain_multimodal_rag = utils.multi_modal_rag_chain(retriever_multi_vector_img)

    user_input = st.text_input("Your question:")
    if user_input:
        docs = retriever_multi_vector_img.invoke(user_input)
        response = chain_multimodal_rag.invoke(user_input)

        st.write("Assistant:", response['answer'])

        for doc in docs:
            if doc in image_summaries:
                index = image_summaries.index(doc)
                st.write("Image Referred from:", utils.plt_img_base64(img_base64_list[index]))
            elif doc in text_summaries:
                st.write("Text Referred from:", doc)
            elif doc in table_summaries:
                st.write("Table Reffered from:", doc)

elif uploaded_file and not apikey:
    st.info("Enter API Key to proceed!")
else:
    st.info("Upload the Document")