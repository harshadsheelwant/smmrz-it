import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import streamlit_shadcn_ui as ui
from streamlit_extras.buy_me_a_coffee import button

checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, offload_folder = 'offload', device_map = 'auto', torch_dtype = torch.float32)



def preprocess(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

def llm_pipeline(filepath):
  pipe_sum = pipeline('summarization',
                      model= base_model,
                      tokenizer=tokenizer,
                      max_length = 500,
                      min_length = 50)
  input_text = preprocess(filepath)
  pdf_summary = pipe_sum(input_text)
  pdf_summary = pdf_summary[0]['summary_text']
  return pdf_summary

def llm_pipeline_notpdf(input_notpdf):
  pipe_sum_notpdf = pipeline('summarization',
                      model= base_model,
                      tokenizer=tokenizer,
                      max_length = 500,
                      min_length = 50)
  notpdf_summary = pipe_sum_notpdf(input_notpdf)
  notpdf_summary = notpdf_summary[0]['summary_text']
  return notpdf_summary

def llm_pipeline_web(extracted_text):
  pipe_sum_web = pipeline('summarization',
                      model= base_model,
                      tokenizer=tokenizer,
                      max_length = 500,
                      min_length = 50)
  web_summary = pipe_sum_web(extracted_text)
  web_summary = web_summary[0]['summary_text']
  return web_summary


@st.cache_data
#function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code
st.set_page_config(
    page_title="Small PDF Summarizer",
    page_icon="📄",
    layout="wide",
)


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
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                pdf_summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                print(pdf_summary)
                st.success(pdf_summary)

    input_notpdf = st.text_input(
        "Enter some text 👇")

    if input_notpdf is not None:

        if ui.button(text="Summarize Text", key="styled_btn_tailwind", class_name="bg-orange-500 text-white"):
            notpdf_summary = llm_pipeline_notpdf(input_notpdf)
            st.info(("Summarization Complete"))
            print(notpdf_summary)
            st.success(notpdf_summary)

    url = st.text_input("Enter the URL of the website 👇")

    if url is not None:

        if ui.button(text="Summarize Website", key="styled_btn_tailwind_2", class_name="bg-orange-500 text-white"):
            extracted_text = extract_text_from_website(url)
            extracted_text = extracted_text[:1500]
            web_summary = llm_pipeline_web(extracted_text)
            st.info(("Summarization Complete"))
            print(web_summary)
            st.success(web_summary)

            st.write("This feature is still under development, it converts the text as it is present in the input website, trucates it and summarizes the text using a LLM model.")


    button(username="harshadsheelwant", floating=False, width=221)                
    ui.link_button(text="My LinkedIN", url="https://www.linkedin.com/in/harshadsheelwant/", key="link_btn1", class_name="bg-black hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")
    ui.link_button(text="My Github", url="https://github.com/harshadsheelwant", key="link_btn2", class_name="bg-black shadow-cyan-500/50 hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")

if __name__ == '__main__':
  main()
