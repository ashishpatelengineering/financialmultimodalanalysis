import os
import time
import tempfile
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import fitz
import PIL.Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_page():
    """Set up the Streamlit page with a financial analysis focus."""
    st.header("AI-Powered Financial Analysis", anchor=False, divider="blue")


def get_media_type():
    """Display a sidebar radio button to select the type of financial data source."""
    st.sidebar.header("Select Financial Data Source Type", divider='orange')
    media_type = st.sidebar.radio(
        "Choose one:",
        ("PDF", "Image", "Video", "Audio")
    )
    return media_type


def get_llm_settings():
    """Display sidebar options for configuring the LLM for financial insights."""
    st.sidebar.header("LLM Configuration for Financial Analysis", divider='rainbow')
    
    model = "gemini-1.5-flash"

    temp_tip = (
        '''
        Lower temperatures ensure more factual and consistent interpretations 
        of financial data, while higher temperatures may introduce creative but 
        less reliable inferences.
        '''
    )
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.25, help=temp_tip
    )

    top_p_tip = (
        "For nucleus sampling: Lower values keep analysis focused on likely scenarios, "
        "higher values consider more speculative possibilities."
    )
    top_p = st.sidebar.slider(
        "Top P:", min_value=0.0, max_value=1.0, value=0.94, step=0.01, help=top_p_tip
    )

    max_tokens_tip = "Maximum length of financial analysis output. Limit: 8194 tokens."
    max_tokens = st.sidebar.slider(
        "Maximum Tokens:", min_value=100, max_value=5000, value=2000, step=100, help=max_tokens_tip
    )

    return model, temperature, top_p, max_tokens


def extract_images_from_pdf(pdf_file):
    """Extract embedded charts or tables from a PDF financial report."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    doc = fitz.open(tmp_file_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(
            matrix=fitz.Identity, dpi=None, colorspace=fitz.csRGB, clip=None, alpha=False, annots=True
        )
        img_path = f"pdfimage-{page.number}.jpg"
        pix.save(img_path)
        images.append(img_path)
    return images


def main():
    """Main function to run the Streamlit app for financial analysis."""
    setup_page()
    media_type = get_media_type()
    model, temperature, top_p, max_tokens = get_llm_settings()

    if media_type == "PDF (Financial Reports)":
        uploaded_files = st.file_uploader("Upload annual reports, earnings releases, or market analysis PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
            )
            question = st.text_input("Enter your financial analysis question (e.g., 'Summarize revenue trends').")
            if question:
                response = model.generate_content([question, text])
                st.write(response.text)

    elif media_type == "Image (Charts/Statements)":
        image_file = st.file_uploader("Upload an image of a stock chart, financial statement, or market diagram", type=["jpg", "jpeg", "png"])
        if image_file:
            image = PIL.Image.open(image_file)
            st.image(image, caption="Uploaded Financial Chart/Image", use_container_width=True)

            prompt = st.text_input("Enter your financial insight request (e.g., 'Analyze this balance sheet').")
            if prompt:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                }
                model = genai.GenerativeModel(model_name=model, generation_config=generation_config)
                response = model.generate_content([image, prompt], request_options={"timeout": 600})
                st.markdown(response.text)

    elif media_type == "Video (Market Briefings)":
        video_file = st.file_uploader("Upload a market briefing or financial news video", type=["mp4"])
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file.read())
                tmp_file_path = tmp_file.name

            st.video(tmp_file_path, format="video/mp4", start_time=0)

            video_file = genai.upload_file(path=tmp_file_path)

            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)

            prompt = st.text_input("Enter your financial question (e.g., 'Summarize key investment recommendations').")
            if prompt:
                model = genai.GenerativeModel(model_name=model)
                response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
                st.markdown(response.text)

                genai.delete_file(video_file.name)
                print(f"Deleted file {video_file.uri}")

    elif media_type == "Audio (Earnings Calls)":
        audio_file = st.file_uploader("Upload an earnings call or investor meeting audio file", type=["mp3", "wav"])
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name

            st.audio(tmp_file_path, format="audio/mp3", start_time=0)

            audio_file = genai.upload_file(path=tmp_file_path)

            while audio_file.state.name == "PROCESSING":
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
                raise ValueError(audio_file.state.name)

            prompt = st.text_input("Enter your financial query (e.g., 'List the main growth strategies discussed').")
            if prompt:
                model = genai.GenerativeModel(model_name=model)
                response = model.generate_content([audio_file, prompt], request_options={"timeout": 600})
                st.markdown(response.text)

                genai.delete_file(audio_file.name)
                print(f"Deleted file {audio_file.uri}")


if __name__ == "__main__":
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY_NEW")
    genai.configure(api_key=GOOGLE_API_KEY)
    main()
