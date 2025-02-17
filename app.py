import os
import time
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
import streamlit as st
from PIL import Image
import io
import logging
import json
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import markdown
import numpy as np
from pydantic import BaseModel, Field
import mimetypes
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit settings
st.set_page_config(
    page_title="Gemini Flash Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .upload-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .results-container {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disable Streamlit usage statistics
if not os.path.exists(".streamlit"):
    os.makedirs(".streamlit")
    
with open(".streamlit/config.toml", "w") as f:
    f.write("[browser]\ngatherUsageStats = false\n")

class ResponseFormat(BaseModel):
    """Model for structured response format"""
    content_type: str = Field(..., description="Type of content (text, table, code, chart, error)")
    data: Any = Field(..., description="The actual content data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the content")

class GeminiProcessor:
    """Handles file processing using Google's Gemini Vision API."""
    
    SUPPORTED_MIME_TYPES = {
        'text/plain': '.txt',
        'text/markdown': '.md',
        'text/csv': '.csv',
        'application/json': '.json',
        'application/pdf': '.pdf',
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': '.png',
        'image/webp': '.webp'
    }
    
    def __init__(self, max_retries: int = 3):
        """Initialize the Gemini processor."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not found")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.max_retries = max_retries
            logger.info("Successfully initialized Gemini processor")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini processor: {str(e)}")
            raise

    def _validate_file(self, file) -> Optional[str]:
        """Validate the uploaded file."""
        try:
            if not file:
                return "No file provided"
            
            # Check file size (20MB limit)
            if file.size > 20 * 1024 * 1024:
                return f"File {file.name} exceeds 20MB size limit"
            
            # Check file type
            mime_type, _ = mimetypes.guess_type(file.name)
            if not mime_type or mime_type not in self.SUPPORTED_MIME_TYPES:
                return f"Unsupported file type: {file.name}. Supported types: {', '.join(self.SUPPORTED_MIME_TYPES.keys())}"
            
            # Additional validation for images
            if mime_type.startswith('image/'):
                try:
                    Image.open(file)
                except Exception:
                    return f"Invalid image file: {file.name}"
            
            return None
            
        except Exception as e:
            return f"Error validating file: {str(e)}"

    def _parse_response(self, response_text: str) -> List[ResponseFormat]:
        """Parse and structure the model's response."""
        try:
            responses = []
            
            # Try to detect if response contains a table
            if '|' in response_text and '-|-' in response_text:
                try:
                    # Convert markdown table to DataFrame
                    lines = [line for line in response_text.split('\n') if line.strip()]
                    header = [col.strip() for col in lines[0].strip('|').split('|')]
                    df = pd.DataFrame([
                        [cell.strip() for cell in line.strip('|').split('|')]
                        for line in lines[2:] if '|' in line and '-|-' not in line
                    ], columns=header)
                    
                    responses.append(ResponseFormat(
                        content_type="table",
                        data=df.to_dict('records'),
                        metadata={"columns": df.columns.tolist()}
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing table: {str(e)}")

            # Try to detect if response contains code
            if '```' in response_text:
                try:
                    code_blocks = []
                    parts = response_text.split('```')
                    
                    for i in range(1, len(parts), 2):
                        lang = parts[i].split('\n')[0].strip()
                        code = '\n'.join(parts[i].split('\n')[1:])
                        code_blocks.append(ResponseFormat(
                            content_type="code",
                            data=code,
                            metadata={"language": lang}
                        ))
                    
                    responses.extend(code_blocks)
                except Exception as e:
                    logger.warning(f"Error parsing code blocks: {str(e)}")

            # If no structured content was detected, return as text
            if not responses:
                responses.append(ResponseFormat(
                    content_type="text",
                    data=response_text,
                    metadata={}
                ))

            return responses

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return [ResponseFormat(
                content_type="error",
                data=str(e),
                metadata={"original_text": response_text}
            )]

    def process_files(self, files: List[Any], prompt: str) -> List[ResponseFormat]:
        """Process multiple files with the given prompt."""
        if not files:
            return [ResponseFormat(
                content_type="error",
                data="No files provided for processing",
                metadata={}
            )]

        results = []
        total_files = len(files)

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(files, 1):
            try:
                # Update progress
                progress = idx / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing file {idx} of {total_files}: {file.name}")

                # Validate file
                validation_error = self._validate_file(file)
                if validation_error:
                    results.append(ResponseFormat(
                        content_type="error",
                        data=validation_error,
                        metadata={"filename": file.name}
                    ))
                    continue

                # Process with retries
                for attempt in range(self.max_retries):
                    try:
                        # Prepare file content
                        content = file.read()
                        file.seek(0)  # Reset file pointer for potential retry
                        
                        # Process with Gemini
                        response = self.model.generate_content([
                            prompt,
                            content
                        ])

                        if response and response.text:
                            parsed_responses = self._parse_response(response.text)
                            for resp in parsed_responses:
                                resp.metadata["filename"] = file.name
                                resp.metadata["processed_at"] = datetime.now().isoformat()
                            results.extend(parsed_responses)
                            break
                        else:
                            raise ValueError("No response from model")

                    except Exception as e:
                        if "API key not valid" in str(e):
                            results.append(ResponseFormat(
                                content_type="error",
                                data="Invalid API key. Please check your GOOGLE_API_KEY in the .env file.",
                                metadata={"filename": file.name}
                            ))
                            break
                        elif attempt == self.max_retries - 1:
                            results.append(ResponseFormat(
                                content_type="error",
                                data=str(e),
                                metadata={"filename": file.name}
                            ))
                        time.sleep(1)  # Wait before retry

            except Exception as e:
                results.append(ResponseFormat(
                    content_type="error",
                    data=f"Unexpected error: {str(e)}",
                    metadata={"filename": file.name}
                ))

        # Clear progress tracking
        progress_bar.empty()
        status_text.empty()

        return results

def save_results(results: List[ResponseFormat], base_dir: str = "results"):
    """Save processing results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(results):
        filename = f"{result.metadata.get('filename', f'result_{idx}')}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "content_type": result.content_type,
                "data": result.data,
                "metadata": result.metadata
            }, f, indent=2)
    
    return save_dir

def display_response(response: ResponseFormat):
    """Display response based on its content type."""
    try:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        if response.content_type == "table":
            st.write(f"### Results for {response.metadata.get('filename', 'Unknown file')}")
            df = pd.DataFrame(response.data)
            st.dataframe(df, use_container_width=True)
            
            # Add download button for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                f"{response.metadata.get('filename', 'results')}.csv",
                "text/csv",
                key=f"download_{response.metadata.get('filename', 'results')}"
            )

        elif response.content_type == "code":
            st.write(f"### Code from {response.metadata.get('filename', 'Unknown file')}")
            st.code(response.data, language=response.metadata.get("language", ""))

        elif response.content_type == "error":
            st.error(f"Error processing {response.metadata.get('filename', 'Unknown file')}: {response.data}")

        else:  # Default text display
            st.write(f"### Results for {response.metadata.get('filename', 'Unknown file')}")
            st.markdown(response.data)
        
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying response: {str(e)}")

def main():
    """Main application function."""
    st.title("🤖 Gemini Flash Chat Interface")
    st.write("Upload files and process them with custom prompts using Gemini Flash 2.0")

    # Initialize session state
    if 'processor' not in st.session_state:
        try:
            st.session_state.processor = GeminiProcessor()
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            return

    # File upload
    st.markdown('<p class="upload-text">Upload Files</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose files to process",
        accept_multiple_files=True,
        help="Upload one or more files to process. Supported types: text, markdown, CSV, JSON, PDF, and images (JPEG, PNG, WebP)"
    )

    # Prompt input
    st.markdown('<p class="upload-text">Enter Prompt</p>', unsafe_allow_html=True)
    prompt = st.text_area(
        "What would you like to do with these files?",
        help="Enter the prompt to process the files with",
        height=100
    )

    # Process button
    if st.button("Process Files", disabled=not (uploaded_files and prompt)):
        if not uploaded_files:
            st.warning("Please upload at least one file")
            return
        
        if not prompt:
            st.warning("Please enter a prompt")
            return

        try:
            with st.spinner("Processing files..."):
                results = st.session_state.processor.process_files(uploaded_files, prompt)
                
                # Save results
                save_dir = save_results(results)
                st.success(f"Results saved to {save_dir}")

                # Display results
                for result in results:
                    display_response(result)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

    # Display usage instructions in sidebar
    with st.sidebar:
        st.markdown("""
        ### How to Use
        1. Upload one or more files using the file uploader
        2. Enter your prompt in the text area
        3. Click "Process Files" to start processing
        4. View results below the form

        ### Supported File Types
        - Text files (.txt)
        - Markdown files (.md)
        - CSV files (.csv)
        - JSON files (.json)
        - PDF files (.pdf)
        - Images:
          - JPEG (.jpg, .jpeg)
          - PNG (.png)
          - WebP (.webp)

        ### Features
        - Multiple file upload
        - Custom prompts
        - Table display with CSV export
        - Code highlighting
        - Error handling
        - Results auto-saving
        """)

if __name__ == "__main__":
    main() 