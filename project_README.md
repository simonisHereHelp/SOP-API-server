# Project End User-Focused [or others] SOP Writer


This multiple-agent APP takes the uploaded Docs Draft, verifies it scopes and apply per the rules and guidelines as established in the RAG base, then generate a Draft SOP.

The App contains following Agents (nodes):

-upload_checker: check the upload is a draft SOP, a draft doc should qualify terms: a) containing one or more identifiable upstream docs, b) containing one or more identifiable downstream docs, c) containing reference of Key Elements [] 
-
-

1. Clone the repository: `git clone https://github.com/bhargobdeka/RAG-chatbot-Speckly.git`


server.py: [venv]python server.py [render] https://sop-api-server.onrender.com
sop-streamlit.py: streamlit run sop-streamlit.py [render] https://sop-streamlit.onrender.com
sop-gradio.py: deployed in HF https://huggingface.co/spaces/hsienchen/sop-gradio

2. LangServe
file_server / file_streamlit: https://github.com/langchain-ai/langserve/tree/main/examples/file_processing

3. review LangGraph Examples
https://github.com/langchain-ai/langgraph/tree/main/examples (official)
https://github.com/mcfatbeard57/LangGraph-RAG-examples (curated by someone)