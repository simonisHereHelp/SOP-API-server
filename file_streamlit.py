import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint
import base64

st.title("PDF Text Extractor")
st.write("Upload a PDF file to extract text from the first page.")

# Input field for uploading a PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing..."):
        try:
            # Read the uploaded PDF file content
            doc_data = uploaded_file.read()
            encoded_data = base64.b64encode(doc_data).decode("utf-8")

            # Initialize the RemoteRunnable
            runnable = RemoteRunnable("http://localhost:8000/pdf")

            # Invoke the runnable and verify the output
            output = runnable.invoke({"file": encoded_data, "num_chars": 100})

            # Extract and display the final output if it exists
            if isinstance(output, str):  # Ensure the output is a dictionary
                st.write("Extracted Text:")
                st.write(output)
            else:
                st.write("Unexpected output format.")

        except Exception as e:
            # Handle any errors that occur during processing
            st.error(f"Error: {e}")
