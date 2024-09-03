import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint
import base64

st.title("PDF Text Extractor and Q&A")
st.write("Upload a PDF file to extract text and ask questions about it.")

# Input field for uploading a PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Input field for specifying the number of characters to extract
num_chars = st.text_input("Number of characters to extract:", value="100")

# Input field for asking a question about the uploaded PDF
question_text = st.text_input("Ask a question about the uploaded PDF:")

# Define the query function
def query():
    try:
        # Check if a file is uploaded
        if uploaded_file is not None:
            with st.spinner("Processing uploaded PDF..."):
                # Read the uploaded PDF file content
                doc_data = uploaded_file.read()
                encoded_data = base64.b64encode(doc_data).decode("utf-8")
        else:
            st.warning("No file uploaded. Proceeding without PDF content.")
            encoded_data = ""  # Use an empty string if no file is uploaded

        # Initialize the RemoteRunnable
        runnable = RemoteRunnable("http://localhost:8000/pdf")

        # Prepare the input dictionary with both encoded data and question
        input_data = {
            "file": encoded_data,
            "num_chars": num_chars,
            "question": question_text
        }

        # Invoke the runnable and verify the output
        output = runnable.invoke(input_data)

        # Extract and display the final output if it exists
        if isinstance(output, str):
            st.write("Response:")
            st.write(output)
        else:
            st.write("Unexpected output format.")

    except Exception as e:
        # Handle any errors that occur during processing
        st.error(f"Error: {e}")

# Add a submit button
if st.button("Submit"):
    query()
