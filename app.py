import gradio as gr
from utils import Translation, NeuralSearcher
from gradio_client import Client
import os
import vecs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

collection_name = "documents"
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = os.getenv("supabase_db")
api_client = Client("eswardivi/Phi-3-mini-128k-instruct")
lan = "en"
vx = vecs.create_client(client)
docs = vx.get_or_create_collection(name=collection_name, dimension=384)

def reply(message, history):
    global docs
    global encoder
    global api_client
    global lan
    txt = Translation(message, "en")
    print(txt.original, lan)
    if txt.original == "en" and lan == "en":
        txt2txt = NeuralSearcher(docs, encoder)
        results = txt2txt.search(message)
        response = api_client.predict(
            f"Context: {results[0][2]['Content']}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        return response
    elif txt.original == "en" and lan != "en":
        txt2txt = NeuralSearcher(docs, encoder)
        transl = Translation(message, lan)
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        response = Translation(response, txt.original)
        return response.translatef()
    elif txt.original != "en" and lan == "en":
        txt2txt = NeuralSearcher(docs, encoder)
        results = txt2txt.search(message)
        transl = Translation(results[0][2]['Content'], "en")
        translation = transl.translatef()
        response = api_client.predict(
            f"Context: {translation}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        t = Translation(response, txt.original)
        res = t.translatef()
        return res
    else:
        txt2txt = NeuralSearcher(docs, encoder)
        transl = Translation(message, lan.replace("\\","").replace("'",""))
        message = transl.translatef()
        results = txt2txt.search(message)
        t = Translation(results[0][2]['Content'], txt.original)
        res = t.translatef()
        response = api_client.predict(
            f"Context: {res}; Prompt: {message}",	# str  in 'Message' Textbox component
            0.4,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
            True,	# bool  in 'Sampling' Checkbox component
            512,	# float (numeric value between 128 and 4096) in 'Max new tokens' Slider component
            api_name="/chat"
        )
        tr = Translation(response, txt.original)
        ress = tr.translatef()
        return ress


demo = gr.ChatInterface(fn=reply, title="Supabase AI Journalist")
demo.launch(server_name="0.0.0.0", share=False)