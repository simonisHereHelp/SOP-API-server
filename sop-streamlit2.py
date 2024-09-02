import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint
import base64

st.title('[company] SOP Server')

# PDF Upload Section
uploaded_file = st.file_uploader("Please upload a valid PDF")

# Text Input Section
input_text = st.text_input('Ask an SOP related question here')

if input_text and uploaded_file is not None:
    with st.spinner("Processing..."):
        try:
            # Read the file content
            file_content = uploaded_file.getvalue()
            file_content_base64 = base64.b64encode(file_content).decode('utf-8')

            # Prepare the input data for the RemoteRunnable
            request_data = {
                "input": input_text,
                "file_name": uploaded_file.name,
                "file_content": file_content_base64
            }
            
            # Create a RemoteRunnable instance and send the request data
            app = RemoteRunnable("http://localhost:8000/speckle_chat/")
            for output in app.stream(request_data):
                for key, value in output.items():
                    pprint(f"Node '{key}':")
                pprint("\n---\n")
            output = value['generation']
            st.write(output)
        
        except Exception as e:
            st.error(f"Error: {e}")
