# importing libraries
# importing mlflow library
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import json
import os
from include.env_values import *
from include.utils import *

# setting up environment
os.environ["OPENAI_API_KEY"] = openai_api_key
with open("./include/config.json") as file:
    config = json.load(file)

# retrieve model from mlflow
model = mlflow.pyfunc.load_model(f"models:/ade_llm_classifier/Production")

# assemble statement input
statement = pd.DataFrame(
    {
        "statement": [
            "There were no other neurological abnormalities and more particularly, no seizures or myocloni were observed."
        ]
    }
)
utils = Utils()
retriever=utils.get_retriever()
st.title("💊 Adverse Drug Affect Classifier")

prompt = st.text_input("Plug in your promt here")
if prompt:
    # assemble statement input
    statement = pd.DataFrame({"statement": [f"{prompt}"]})
    # get response
    output = model.predict(statement)

    st.write(output)
    docs = retriever.get_relevant_documents(prompt)
    
    context = ""
    for doc in docs:
        # get document text
        context = context + "\n" + doc.page_content + "\n" + "###"

    input_prompt = f"""
    understand the statements for any adverse events and predict the [nature]. 'is_ADE' means [statement] reports an adverse event medically and 'not_ADE' means not adverse event.

    {context} \n
    [statement]: {prompt} \n
    [nature]: ''\n

    '###' means end of line.

    return output for last 'statement' in this way:
    [nature]: 'is_ADE' (if adverse avtivity present)
    [nature]: 'not_ADE' (if adverset avtivity not present)
    [nature]: 'I can't Identify'
    """
    with st.expander('Generated Input Prompt'):
        st.code(input_prompt) 