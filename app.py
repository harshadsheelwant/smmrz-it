import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import streamlit_shadcn_ui as ui
from streamlit_extras.buy_me_a_coffee import button
from annotated_text import annotated_text, annotation
import os

checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, offload_folder='offload', device_map='auto', torch_dtype=torch.float32)

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join([text.page_content for text in texts])
    return final_texts

def llm_pipeline(file_path):
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    input_text = file_preprocessing(file_path)
    pdf_summary = pipe_sum(input_text)
    return pdf_summary[0]['summary_text']

def llm_pipeline_notpdf(input_notpdf):
    pipe_sum_notpdf = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    notpdf_summary = pipe_sum_notpdf(input_notpdf)
    return notpdf_summary[0]['summary_text']

def llm_pipeline_web(extracted_text):
    pipe_sum_web = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    web_summary = pipe_sum_web(extracted_text)
    return web_summary[0]['summary_text']

@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.components.v1.html(pdf_display, width=800, height=600)

st.set_page_config(page_title="Summarize-It", page_icon="📄", layout="wide")

def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    web_text = soup.get_text(separator=' ')
    web_text = "\n".join(line for line in web_text.splitlines() if line.strip())
    return web_text

def main():
    st.title("Summarize-It📄")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if ui.button(text="Summarize PDF", key="styled_btn_tailwind_1", class_name="bg-orange-500 text-white"):
            col1, col2 = st.columns(2)
            file_path = os.path.join("tempDir", uploaded_file.name)
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                displayPDF(file_path)

            with col2:
                pdf_summary = llm_pipeline(file_path)
                st.info("Summarization Complete")
                st.success(pdf_summary)

    input_notpdf = st.text_input("Enter some text 👇")

    if input_notpdf:
        if ui.button(text="Summarize Text", key="styled_btn_tailwind", class_name="bg-orange-500 text-white"):
            notpdf_summary = llm_pipeline_notpdf(input_notpdf)
            st.info("Summarization Complete")
            st.success(notpdf_summary)

    url = st.text_input("Enter the URL of the website 👇")

    if url:
        if ui.button(text="Summarize Website", key="styled_btn_tailwind_2", class_name="bg-orange-500 text-white"):
            extracted_text = extract_text_from_website(url)
            extracted_text = extracted_text[:1500]
            web_summary = llm_pipeline_web(extracted_text)
            st.info("Summarization Complete")
            st.success(web_summary)

            st.markdown(
                """
                This feature is still under development. It converts the text as it is present in the input website, truncates it, and summarizes the text using an 'LLM model'.
                """
            )

    annotated_text("The summarization by this application is done using ", annotation("LaMini-Flan-T5-248M", "Summarization Pipeline", color="#98e2cf", border="1px dashed red"))
    button(username="harshadsheelwant", floating=False, width=221)
    ui.link_button(text="My LinkedIN", url="https://www.linkedin.com/in/harshadsheelwant/", key="link_btn1", class_name="bg-black hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")
    ui.link_button(text="My Github", url="https://github.com/harshadsheelwant", key="link_btn2", class_name="bg-black shadow-cyan-500/50 hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")

if __name__ == '__main__':
    os.makedirs("tempDir", exist_ok=True)
    main()
