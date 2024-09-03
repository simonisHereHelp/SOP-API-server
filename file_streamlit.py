import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint
import base64

st.title("PDF Text Extractor and Q&A")
st.write("Upload one or more PDF files to extract text and ask questions about them.")

# Input field for uploading multiple PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Input field for specifying the number of characters to extract
num_chars = st.text_input("Number of characters to extract from each file:", value="500")

# Input field for asking a question about the uploaded PDFs
question_text = st.text_input("Ask a question about the uploaded PDFs:")

# Define the query function
def query():
    if uploaded_files:
        encoded_files = []
        for file in uploaded_files:
            doc_data = file.read()
            encoded_data = base64.b64encode(doc_data).decode("utf-8")
            encoded_files.append(encoded_data)

        with st.spinner("Processing uploaded PDFs..."):
            try:
                # Initialize the RemoteRunnable
                runnable = RemoteRunnable("http://localhost:8000/pdf")

                # Prepare the input dictionary with both encoded data and question
                input_data = {
                    "files": encoded_files,
                    "num_chars": num_chars,
                    "question": question_text
                }

                # Invoke the runnable and verify the output
                output = runnable.invoke(input_data)

                # Extract and display the final output if it exists
                if isinstance(output, list):
                    for i, result in enumerate(output):
                        st.write(f"Response for File {i+1}:")
                        st.write(result)
                else:
                    st.write("Unexpected output format.")

            except Exception as e:
                # Handle any errors that occur during processing
                st.error(f"Error: {e}")
    else:
        st.warning("No PDF files uploaded.")

# Add a submit button
if st.button("Submit"):
    query()
