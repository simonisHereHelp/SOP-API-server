import gradio as gr
from langserve import RemoteRunnable
from pprint import pprint

def get_response(input_text):
    app = RemoteRunnable("https://sop-api-server.onrender.com/speckle_chat/")
    for output in app.stream({"input": input_text}):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    output = value['generation']
    return output   

# Create the UI In Gradio
iface = gr.Interface(fn=get_response, 
          inputs=gr.Textbox(
          value="Enter your question"), 
          outputs="textbox",  
          title="Q&A over SOP docs",
          description="Ask a question about SOP docs and get an answer from the AI assistant. This assistant looks up relevant documents and answers your code-related question.",
          examples=[["How to fill out Form 3602A?"], 
                  ["Give out cases for VR-based surgical procedures."],["Elaborate cases or indications for VR-based rehabiliations."]
                  ],
          theme=gr.themes.Soft(),
          allow_flagging="never",)

iface.launch(share=True) # put share equal to True for public URL