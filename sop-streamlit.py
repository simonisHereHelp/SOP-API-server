import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint

st.title('[company] SOP Server')
input_text = st.text_input('ask SOP related question here')

if input_text:
    with st.spinner("Processing..."):
        try:
            app = RemoteRunnable("https://localhost:8000/speckle_chat/")
            for output in app.stream({"input": input_text}):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")
            output = value['generation']  
            st.write(output)
        
        except Exception as e:
            st.error(f"Error: {e}")
        

