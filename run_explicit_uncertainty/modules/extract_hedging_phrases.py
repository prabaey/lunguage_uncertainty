from google import genai
from google.genai import types

import json
import pandas as pd

SYSTEM_PROMPT = """You are a radiologist who is given pairs of entities and sentences. Each entity appears in the corresponding sentence. 
Your task is to identify and extract only the words or phrases in the sentence that express uncertainty specifically about the given entity."""

USER_PROMPT = """TASK: You are given pairs of entities and sentences. Each entity appears in the corresponding sentence. 
Your task is to identify and extract only the words or phrases in the sentence that express uncertainty specifically about the given entity.

Important notes:
- The sentence may mention multiple entities, but you should extract uncertainty clues only for the specified entity. Please refer to the examples below that show how you should handle such cases.
- Look for words or phrases that suggest uncertainty, speculation, approximation, possibility, or lack of definitiveness (e.g., "might", "possibly", "suggests", "appears to", "in some cases").
- Return a list of such uncertainty clues found in the sentence and relevant to the query entity.

Return your output as a list: ["<word or phrase 1>", "<word or phrase 2>", ...] 
If there are no uncertainty clues related to the given entity, return an empty list.

Below are 10 examples, after which you must complete the task for an unseen query. 

INPUT:
{{
	"entity": "pulmonary edema",
	"sentence": "overall, however, there is a more focal airspace opacity in the left mid and lower lung, which may reflect asymmetric pulmonary edema or an infectious process, less likely atelectasis."
}}
OUTPUT: 
["may", "or"]

INPUT:
{{
	"entity": "infectious process",
	"sentence": "overall, however, there is a more focal airspace opacity in the left mid and lower lung, which may reflect asymmetric pulmonary edema or an infectious process, less likely atelectasis."
}}
OUTPUT: 
["may", "or"]

INPUT:
{{
	"entity": "atelectasis",
	"sentence": "overall, however, there is a more focal airspace opacity in the left mid and lower lung, which may reflect asymmetric pulmonary edema or an infectious process, less likely atelectasis."
}}
OUTPUT: 
["less likely"]

INPUT:
{{
	"entity": "prominence",
	"sentence": "cardiac and mediastinal silhouettes are stable with possible slight decrease in right paratracheal prominence."
}}
OUTPUT: 
["possible"]

INPUT:
{{
	"entity": "pneumonia",
	"sentence": "given the clinical presentation, pneumonia must be suspected."
}}
OUTPUT: 
["suspected"]

INPUT:
{{
	"entity": "pneumonia",
	"sentence": "bronchial wall thickening or peribronchial infiltration in the lower lungs where most pronounced bronchiectasis is have worsened since consistent either with a flare of bronchiectasis or development of peribronchial pneumonia."
}}
OUTPUT: 
["either", "or"]

INPUT:
{{
	"entity": "bronchiectasis",
	"sentence": "right basilar opacification may reflect bronchiectasis, though infection cannot be completely excluded."
}}
OUTPUT: 
["may"]

INPUT:
{{
	"entity": "infection",
	"sentence": "right basilar opacification may reflect bronchiectasis, though infection cannot be completely excluded."
}}
OUTPUT: 
["cannot be completely excluded"]

INPUT:
{{
	"entity": "amiodarone toxicity",
	"sentence": "differential for these lesions includes amiodarone toxicity and cryptogenic organizing pneumonia."
}}
OUTPUT: 
["differential"]

INPUT:
{{
	"entity": "pleural effusion",
	"sentence": "a left pleural effusion and atelectasis obscure the left cardiac and hemidiaphragmatic contours more than the prior day."
}}
OUTPUT: 
[]

Examine the entity and sentence pair below. If the sentence also talks about other entities, first identify the part of the 
sentence that is talking about the query entity, and then extract phrases expressing uncertainty specifically related to that entity.
Return your output as a list: ["<word or phrase 1>", "<word or phrase 2>", ...] 
If there are no uncertainty clues related to the given entity, return an empty list.

INPUT: 
{{
    "entity": {entity}
    "sentence": {sentence}
}}
OUTPUT:
"""

def prompt_gemini_uncertainty_phrases(entity, sentence, client):

    model = "gemini-2.5-flash"
    thinking_config = types.ThinkingConfig(thinking_budget=1000)
    max_output_tokens = 1100

    prompt_string = USER_PROMPT.format(entity=entity, sentence=sentence)
    system_prompt = SYSTEM_PROMPT

    contents = [
        types.Content(
            role="user",
            parts=[
            types.Part.from_text(text = prompt_string)
            ]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1, 
        top_p = 1, 
        seed = 2025,
        max_output_tokens = max_output_tokens, 
        system_instruction = [types.Part.from_text(text=system_prompt)],
        thinking_config=thinking_config
    )

    response = client.models.generate_content(
        model=model,
        contents = contents, 
        config = generate_content_config
    )

    return prompt_string, system_prompt, response.text.strip()

def extract_uncertainty_phrases():
    """
    Extract hedging phrases for each entity-sentence pair in the Lunguage dataset.
    """

    client = genai.Client(
        vertexai=True,
        project="your-project-here",
        location="global",
    )

    gold_dataset_path = '../data_resources/lunguage/Lunguage.csv'
    df_gold = pd.read_csv(gold_dataset_path)
    df_tent = df_gold[df_gold["dx_certainty"] == "tentative"].copy()
    
    output_file = "../data_resources/hedging_phrase_extracted.jsonl"
    n_done = 0

    with open(output_file, "a", encoding="utf-8") as f:
        for i, row in df_tent.iterrows(): 
            
            idx = row.name
            study_id = row["study_id"]
            section = row["section"]
            ent_idx = row["ent_idx"]
            entity_name = row["ent"]
            sentence = row["sent"]
            
            _, _, response = prompt_gemini_uncertainty_phrases(entity_name, sentence, client)
            if "OUTPUT" in response: 
                response = response[len("OUTPUT:"):].strip()
            try: 
                resp_list = eval(response)
            except SyntaxError as se: 
                print(f"syntax error for idx {idx}, response {response}. saving response as-is")
                resp_list = response
            
            result = {
                    "idx": idx,
                    "study_id": study_id,
                    "section": section,
                    "ent_idx": ent_idx,
                    "entity_name": entity_name,
                    "sentence": sentence,
                    "resp_list": resp_list
                }
            
            f.write(json.dumps(result) + "\n")
            f.flush()

            n_done += 1
            if n_done % 100 == 0: 
                print(n_done)