import os
import io
import streamlit as st
import pypdf
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

# ======================  Load Environment Variables  ======================
load_dotenv()
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# ======================  Streamlit UI Setup  ======================
st.set_page_config(page_title='AI Resume Analyse', page_icon='📄', layout='centered')
st.title('AI Resume')
st.markdown('Upload your resume and get **AI-powered feedback**.')

upload_file = st.file_uploader('Upload your Resume', type=['pdf', 'txt'])
job_role = st.text_input('Enter the Role you are targeting (optional):')
analyse = st.button('Analyse')

# ======================  File Content Extraction  ======================

def pdf_content(upload_file):
    pdf_reader = pypdf.PdfReader(upload_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text

def pdf_txt_content(upload_file):
    if upload_file.type == 'application/pdf':
        return pdf_content(io.BytesIO(upload_file.read()))
    return upload_file.read().decode('utf-8')

# ======================  LLM Analysis  ======================

if analyse and upload_file:
    try:
        file_content = pdf_txt_content(upload_file)

        if not file_content.strip():
            st.error('There is no content in your resume.')
            st.stop()

        prompt = f"""
        Analyse this resume and provide detailed feedback.
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skill presentation
        3. Experience description
        4. Specific improvements for {job_role if job_role else 'a general job application'}

        Resume content:
        {file_content}

        Please provide the analysis in a clear, structured format with specific recommendations.
        """

        llm = HuggingFaceEndpoint(
            repo_id='HuggingFaceH4/zephyr-7b-beta',  
            task='text-generation',
            huggingfacehub_api_token=token
        )

        chat_model = ChatHuggingFace(llm=llm)

        response = chat_model.invoke([
            SystemMessage(content="You are an expert resume reviewer with years of experience."),
            HumanMessage(content=prompt)
        ])

        st.markdown('### 🧠 Analysis Result')
        st.write(response.content)

    except Exception as e:
        st.error(f'An error has occurred: {str(e)}')
