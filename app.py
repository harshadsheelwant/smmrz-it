import streamlit as st
import os
import re
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import torch
import base64
import streamlit_shadcn_ui as ui
from streamlit_extras.buy_me_a_coffee import button
from annotated_text import annotated_text, annotation
from streamlit_pdf_viewer import pdf_viewer
import whisper


checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)


def transcribe_video(uploaded_video):
    model = whisper.load_model("base")
    result=model.transcribe("video.mp4")
    transcription = result["text"]


def get_transcript(yt_url):
    try:
        video_id = yt_url.split("v=")[-1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception as e:
        st.error(f"Error fetching transcript list for video ID {video_id}: {e}")
        return ""
    
    try:
        # Find manually created transcript if available
        transcript = transcript_list.find_manually_created_transcript()
    except Exception:
        try:
            # Find generated transcripts
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            if not generated_transcripts:
                st.error("No generated transcripts available.")
                return ""
            transcript = generated_transcripts[0]
        except Exception as e:
            st.error(f"No suitable transcript found: {e}")
            return ""
    
    try:
        # Fetch the English translation of the transcript
        english_transcript = transcript.translate('en')
        transcript_for_translation = " ".join([part['text'] for part in english_transcript.fetch()])
    except Exception as e:
        st.error(f"Error fetching English translated transcript text: {e}")
        return ""
    
    return transcript_for_translation





def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

def llm_pipeline(input_text):
  pipe_sum = pipeline('summarization',
                      model= base_model,
                      tokenizer=tokenizer,
                      max_length = 5000,
                      min_length = 50)
  summary = pipe_sum(input_text)
  summary = summary[0]['summary_text']
  return summary



#@st.cache_data
#function to display the PDF of a given file
def displayPDF(file):
#     # Opening file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)
    with open(file, "rb") as f:
         pdf_viewer(f.read(), height=600, width=800)

#streamlit code
st.set_page_config(
    page_title="Summarize-It",
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
    
    st.write("Summarize Video")
    uploaded_video = st.file_uploader("Upload your Video(MP4) file", type=['video/mp4'])

    if uploaded_video is not None:
        if ui.button(text="Summarize Video", key="styled_btn_tailwind_vdo", class_name="bg-orange-500 text-white"):
            col1, col2 = st.columns(2)
            video = "data/"+uploaded_file.name
            # with open(filepath, "wb") as temp_file:
            #     temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded Video")
                st.video(video, format="video/mp4")
                input_text = transcribe_video(video)
                input_text = input_text[:5000]

            with col2:
                vdo_summary = llm_pipeline(input_text)
                st.info("Summarization Complete")
                print(vdo_summary)
                st.success(vdo_summary)

    yt_url=st.text_input('Enter YouTube URL 👇')

    

    if yt_url is not None:

        if ui.button(text="Summarize YouTube Video", key="styled_btn_tailwind_yt", class_name="bg-orange-500 text-white"):
            extracted_transcript = get_transcript(yt_url)
            input_text = extracted_transcript
            col1, col2 = st.columns(2)
            with col1:    
                st.video(yt_url)

            with col2:
                yt_summary = llm_pipeline(input_text)
                st.info("Summarization Complete")
                print(yt_summary)
                st.success(yt_summary)




    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if ui.button(text="Summarize PDF", key="styled_btn_tailwind_pdf", class_name="bg-orange-500 text-white"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)
                input_text = file_preprocessing(filepath)
                input_text = input_text[:5000]

            with col2:
                pdf_summary = llm_pipeline(input_text)
                st.info("Summarization Complete")
                print(pdf_summary)
                st.success(pdf_summary)

    input_text = st.text_input(
        "Enter some text 👇")

    if input_text is not None:

        if ui.button(text="Summarize Text", key="styled_btn_tailwind_txt", class_name="bg-orange-500 text-white"):
            notpdf_summary = llm_pipeline(input_text)
            st.info(("Summarization Complete"))
            print(notpdf_summary)
            st.success(notpdf_summary)

    url = st.text_input("Enter the URL of the website 👇")

    if url is not None:

        if ui.button(text="Summarize Website", key="styled_btn_tailwind_web", class_name="bg-orange-500 text-white"):
            extracted_text = extract_text_from_website(url)
            extracted_text = extracted_text[:1500]
            input_text = extracted_text
            web_summary = llm_pipeline(input_text)
            st.info(("Summarization Complete"))
            print(web_summary)
            st.success(web_summary)

            st.markdown(
                """
                This feature is still under development, it converts the text as it is present in the input website, truncates it to a length acceptable by the used model and summarizes the text using a 'LLM model'.
                """
            )

    annotated_text("The summarization by this application is done using ", annotation("[LaMini-Flan-T5-248M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M)", " Summarization Pipeline", color="#98e2cf", border="1px dashed red"))
    button(username="harshadsheelwant", floating=False, width=221)                
    ui.link_button(text="My LinkedIN", url="https://www.linkedin.com/in/harshadsheelwant/", key="link_btn1", class_name="bg-black hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")
    ui.link_button(text="My Github", url="https://github.com/harshadsheelwant", key="link_btn2", class_name="bg-black shadow-cyan-500/50 hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")




if __name__ == '__main__':
  main()
