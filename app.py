import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import docx
from fpdf import FPDF
import plotly.express as px
from io import BytesIO
import time
import re
from langdetect import detect

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

@st.cache_data
def analyze_content_with_gemini(text, analysis_type="general", language="english"):
    """Analyze the content using Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    
    prompts = {
        "general": f"Analyze this {language} text and provide key insights, main themes, and important points:",
        "summary": f"Provide a comprehensive summary of this {language} text:",
        "key_points": f"Extract and list the key points from this {language} text:",
        "recommendations": f"Based on this {language} text, what are the main recommendations or action items?",
        "sentiment": f"Analyze the sentiment and tone of this {language} text. Provide examples to support your analysis.",
        "topics": f"Identify and analyze the main topics discussed in this {language} text:",
        "keywords": f"Extract the most important keywords and phrases from this {language} text:"
    }

    try:
        response = model.generate_content(f"{prompts.get(analysis_type, prompts['general'])} \n\n{text[:15000]}")
        return response.text
    except Exception as e:
        st.error(f"Error analyzing content: {str(e)}")
        return None

def generate_text_statistics(text):
    """Generate basic text statistics."""
    words = word_tokenize(text)
    sentences = text.split('.')
    
    stats = {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Average Words per Sentence": round(len(words) / len(sentences), 2),
        "Character Count": len(text),
        "Estimated Reading Time (minutes)": round(len(words) / 200, 2)  # Assuming 200 words per minute
    }
    return stats

def create_word_cloud(text):
    """Generate a word cloud from text."""
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         stopwords=stop_words,
                         min_font_size=10).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def export_as_pdf(content, filename="analysis.pdf"):
    """Export analysis as PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split content into lines and add to PDF
    lines = content.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, txt=line, align='L')
    
    return pdf.output(dest='S').encode('latin-1')

def export_as_docx(content, filename="analysis.docx"):
    """Export analysis as DOCX."""
    doc = docx.Document()
    doc.add_paragraph(content)
    
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def detect_language(text):
    """Detect the language of the text."""
    try:
        return detect(text)
    except:
        return "english"

def main():
    st.set_page_config(
        page_title="Enhanced PDF Analyzer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button { width: 100%; }
        .analysis-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 Enhanced PDF Analyzer")
    st.write("Upload PDF files for comprehensive analysis using AI and advanced text analytics.")

    # File upload - multiple files
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Analysis options
        st.sidebar.subheader("Analysis Settings")
        analysis_type = st.sidebar.selectbox(
            "Select analysis type:",
            ["general", "summary", "key_points", "recommendations", "sentiment", "topics", "keywords"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        show_wordcloud = st.sidebar.checkbox("Show Word Cloud", True)
        show_stats = st.sidebar.checkbox("Show Text Statistics", True)
        
        # Process each file
        for uploaded_file in uploaded_files:
            st.subheader(f"Analysis for: {uploaded_file.name}")
            
            # Extract and analyze text
            with st.spinner('Processing PDF...'):
                text = extract_text_from_pdf(uploaded_file)
                
                if text:
                    # Detect language
                    language = detect_language(text)
                    st.info(f"Detected language: {language}")
                    
                    # Create tabs for different analyses
                    tabs = st.tabs(["Analysis", "Statistics", "Visualizations", "Export"])
                    
                    # Analysis tab
                    with tabs[0]:
                        with st.spinner('Analyzing content...'):
                            result = analyze_content_with_gemini(text, analysis_type, language)
                            if result:
                                st.markdown(f"""
                                    <div class="analysis-box">
                                        {result}
                                    </div>
                                """, unsafe_allow_html=True)
                    
                    # Statistics tab
                    with tabs[1]:
                        if show_stats:
                            stats = generate_text_statistics(text)
                            st.subheader("Text Statistics")
                            for key, value in stats.items():
                                st.metric(key, value)
                    
                    # Visualizations tab
                    with tabs[2]:
                        if show_wordcloud:
                            st.subheader("Word Cloud")
                            wordcloud_fig = create_word_cloud(text)
                            st.pyplot(wordcloud_fig)
                    
                    # Export tab
                    with tabs[3]:
                        st.subheader("Export Options")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Download as TXT",
                                data=result,
                                file_name=f"analysis_{analysis_type}.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            pdf_output = export_as_pdf(result)
                            st.download_button(
                                label="Download as PDF",
                                data=pdf_output,
                                file_name=f"analysis_{analysis_type}.pdf",
                                mime="application/pdf"
                            )
                        
                        with col3:
                            docx_output = export_as_docx(result)
                            st.download_button(
                                label="Download as DOCX",
                                data=docx_output.getvalue(),
                                file_name=f"analysis_{analysis_type}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )

    # Sidebar information
    with st.sidebar:
        st.subheader("Features")
        st.write("""
        - Multiple PDF Analysis
        - Language Detection
        - Text Statistics
        - Word Cloud Visualization
        - Multiple Export Formats
        - Various Analysis Types
        """)
        
        st.divider()
        st.markdown("Made with ❤️ using Streamlit and Google Gemini")

if __name__ == "__main__":
    main()