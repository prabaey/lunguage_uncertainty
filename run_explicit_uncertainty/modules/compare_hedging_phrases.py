import os
import httpx
import time
import random
import pandas as pd
import json

from itertools import combinations
from typing import Dict, List, Tuple

from google import genai
from google.genai import types
from openai import AzureOpenAI
import anthropic

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMSentenceComparer:

    def __init__(self, gemini_project=None, openai_endpoint=None, openai_key=None, anthropic_key=None, huggingface_token=None):

        # LLM settings
        self.system_prompt = "You are a radiologist who is ranking sentences expressing uncertainty."

        # Set up http client to avoid socket errors
        self.httpx_client = httpx.Client()

        # Set up Gemini
        self.gemini_client = genai.Client(
            vertexai=True,
            project=gemini_project,
            location="global"
        )

        # Set up OpenAI
        self.endpoint = os.getenv("ENDPOINT_URL", openai_endpoint)
        self.api_key_openai = os.getenv("AZURE_OPENAI_API_KEY", openai_key)
        self.gpt_client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key_openai,
            api_version="2025-01-01-preview",
            http_client = self.httpx_client
        )
        
        # Set up Anthropic
        self.api_key_anthropic = anthropic_key
        self.claude_client = anthropic.Anthropic(
            api_key=self.api_key_anthropic,
            http_client=self.httpx_client
        )

        # Set up MedGemma
        login(token=huggingface_token)
        model_id = "google/medgemma-27b-text-it"
        self.medgemma_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir="./cache",
            torch_dtype="auto",
            device_map="auto",
        )
        self.medgemma_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./cache")
        print("MedGemma model has been downloaded")
        
    def build_prompt_sentence_comparison(self, sentence_1: str, sentence_2: str) -> str:
        return f'''
You will be given two sentences from radiology reports. Each sentence contains a placeholder "<finding>", which represents a medical observation (e.g., consolidation, effusion, nodule). Each sentence includes a phrase that expresses the degree of certainty about the presence or absence of the finding.
Assume there is a certainty spectrum ranging from:

    "<finding> is certainly absent"
    to
    "<finding> is certainly present"

Your task is to identify which sentence is **closer to "<finding> is certainly present"** on this scale, using the context of the sentence. 
In other words, your task is to identify which sentence expresses a higher degree of certainty that the finding is present.

Respond with **only** the chosen sentence (sentence_1 or sentence_2).

Here are some examples.

---

Example 1:
INPUT:
{{
  "sentence_1": "interstitial markings are prominent, suggest possible mild <finding>.",
  "sentence_2": "allowing for low inspiratory volumes, the <finding> is probably unchanged."
}}
OUTPUT:
"sentence_2"

---

Example 2:
INPUT:
{{
  "sentence_1": "given the clinical presentation, <finding> must be suspected.",
  "sentence_2": "although this could represent severe <finding>, the possibility of supervening pneumonia or even developing ards must be considered."
}}
OUTPUT:
"sentence_1"

---

Example 3:
INPUT:
{{
  "sentence_1": "this could be either pneumonia in the left upper lobe or fissural <finding>.",
  "sentence_2": "the presence of a minimal left <finding> cannot be excluded, given blunting of the left costophrenic sinus.",
}}
OUTPUT:
"sentence_1"

---

INPUT:
{{
  "sentence_1": "{sentence_1}",
  "sentence_2": "{sentence_2}"
}}

Which of the two sentences ("sentence_1" or "sentence_2") indicates that <finding> is more certainly present? Respond with your choice **only**.

OUTPUT:
'''.strip()
    

    def query_gemini(self, prompt: str) -> str:
        
        contents = [
            types.Content(
                role="user",
                parts=[
                types.Part.from_text(text = prompt)
                ]
            ),
        ]

        model = "gemini-2.5-flash"
        thinking_config = types.ThinkingConfig(thinking_budget=0)
        max_output_tokens = 100

        generate_content_config = types.GenerateContentConfig(
            temperature = 1, 
            top_p = 1, 
            max_output_tokens = max_output_tokens, 
            system_instruction = [types.Part.from_text(text=self.system_prompt)],
            thinking_config=thinking_config
        )

        response = self.gemini_client.models.generate_content(
            model=model,
            contents = contents, 
            config = generate_content_config
        )

        # Calculate cost
        input_price = 0.3
        output_price = 2.5
        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        except AttributeError:
            input_tokens = 0
            output_tokens = 0
        cost = input_tokens/1000000*input_price + output_tokens/1000000*output_price

        try: 
            return response.text.strip(), cost
        except: 
            print(f"Nonetype returned. Store response as-is.")
            return response.text, cost
        
    def query_gpt4(self, prompt: str) -> str:

        model = "gpt-4o"

        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
        ]

        # Generate the completion
        completion = self.gpt_client.chat.completions.create(
            model=model,
            messages=chat_prompt,
            max_tokens=100,
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        # Calculate cost
        input_price = 2.75
        output_price = 11
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens/1000000*input_price + output_tokens/1000000*output_price

        return completion.choices[0].message.content.strip(), cost
    
    
    def query_claude(self, prompt: str) -> str:

        model = "claude-sonnet-4-20250514"

        chat_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        message = self.claude_client.messages.create(
            model=model,
            max_tokens=100,
            temperature=1,
            system=self.system_prompt,
            messages=chat_prompt
        )

        # Calculate cost
        input_price = 3
        output_price = 15
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = input_tokens/1000000*input_price + output_tokens/1000000*output_price

        return message.content[0].text.strip(), cost
    
    def query_medgemma(self, prompt: str) -> str:

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.medgemma_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.medgemma_model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.medgemma_model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=1, top_p=1)
            generation = generation[0][input_len:]

        decoded = self.medgemma_tokenizer.decode(generation, skip_special_tokens=True)
        cost = 0

        return decoded.strip(), cost


    def compare_sentences(self, sentence_1: str, sentence_2: str, llm_name: str) -> str:

        # use single-sentence prompt
        prompt = self.build_prompt_sentence_comparison(sentence_1, sentence_2)

        if llm_name == "Gemini":
            query_function = self.query_gemini
        elif llm_name == "GPT": 
            query_function = self.query_gpt4
        elif llm_name == "Claude": 
            query_function = self.query_claude
        elif llm_name == "MedGemma":
            query_function = self.query_medgemma
        else: 
            print("Incorrect LLM name. Proceed with Gemini.")
            llm_name = "Gemini"

        response, cost = query_function(prompt)

        return prompt, response, cost
    
class UncertaintyPhrase:
    """ 
    Stores an UncertaintyPhrase object, which is made up of the phrase itself and a set of example sentences where the phrase is used.
    """
    def __init__(self, phrase: str, examples: List[Dict[str, str]]):
        self.phrase = phrase # name of phrase
        self.examples = examples # example sentences

    def sample_sentence(self) -> Dict[str, str]:
        return random.choice(self.examples)


class UncertaintyComparer:
    """ 
    Runs comparisons of phrases by randomly selecting example sentences for each phrase, and calling an LLM to choose a winning phrase in each match. 
    """
    def __init__(self, llm_name, repeat, vocab_data: Dict[str, List[Dict[str, str]]], log_path: str, seed: int=42, sentence_pair_file: str=None):
        
        self.llm_name = llm_name
        self.phrases = {phrase: UncertaintyPhrase(phrase, data) for phrase, data in vocab_data.items()} # list of phrases
        self.seen_pairs = set() # paired sentences which were already tested
        self.log_path = log_path
        self.repeat = repeat
        random.seed(seed)

        # precompute all unique phrase pairs
        pairs = list(combinations(self.phrases.keys(), 2))
        try:
            df_done = pd.read_json(log_path, lines=True)
            self.pairs_to_compare = []
            for pair in pairs: # only do pairs that were not done already
                if len(df_done[(df_done["phrase_1"] == pair[0]) & (df_done["phrase_2"] == pair[1])]) == 0:
                    self.pairs_to_compare.append(pair)
        except:
            self.pairs_to_compare = pairs

        # get pairs of sentences from previous generation file, if provided
        self.pair_to_sentences = {}
        if sentence_pair_file:
            with open(sentence_pair_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    key = (data["phrase_1"], data["phrase_2"])
                    sentence_pair = {"sentence_1": data["sentence_1"], 
                                     "sentence_2": data["sentence_2"], 
                                     "entity_1": data["entity_1"], 
                                     "entity_2": data["entity_2"]}
                    self.pair_to_sentences.setdefault(key, []).append(sentence_pair)
        self.pair_index_tracker = {key: 0 for key in self.pair_to_sentences} # keep track of sentence pairs you've called
        

    def mask_entity(self, sentence: str, entity: str) -> str:
        
        # replace the entity with the neutral <finding> task, in other to make the comparisons agnostic to the particular entities
        return sentence.replace(entity, "<finding>")

    def run_comparison(self, pair: Tuple[str, str], llm_prompter: LLMSentenceComparer):
        """ 
        Runs a comparison for a pair of uncertainty phrases, by prompting the LLM.
        - pair: pair of uncertainty phrases for which to run the comparison
        - llm_prompter: LLMSentenceComparer class which prompts the LLM to make the comparison
        - repeat: how many times the prompting should be repeated (with different sentences randomly sampled with each call)
        """

        # select phrases
        phrase1, phrase2 = pair
        p1 = self.phrases[phrase1]
        p2 = self.phrases[phrase2]

        # get example sentences
        for _ in range(self.repeat):

            sent1, sent2 = [], []

            # if we already have a list of example sentence pairs, then go through them one by one
            if hasattr(self, "pair_to_sentences") and pair in self.pair_to_sentences:

                # keep track of sentence pairs you've already used
                index = self.pair_index_tracker[pair]

                # Make sure there are enough pairs
                if index >= len(self.pair_to_sentences[pair]):
                    raise ValueError(f"Ran out of sentence pairs for phrase pair {pair}")

                # get example sentences
                sent_entity_pair = self.pair_to_sentences[pair][index]
                self.pair_index_tracker[pair] += 1

                # mask the entity
                sent1.append(sent_entity_pair["sentence_1"])
                sent2.append(sent_entity_pair["sentence_2"])

                # keep track of entities for logging
                ent1 = sent_entity_pair["entity_1"]
                ent2 = sent_entity_pair["entity_2"]

            # randomly sample pair of sentences
            else:

                # randomly sample example sentences where the phrase is used
                ex1 = p1.sample_sentence()
                ex2 = p2.sample_sentence()

                # mask the entity with agnostic <finding> tag 
                sent1_masked = self.mask_entity(ex1['sentence'], ex1['entity'])
                sent2_masked = self.mask_entity(ex2['sentence'], ex2['entity'])

                # add to list of sentences
                sent1.append(sent1_masked)
                sent2.append(sent2_masked)

                # keep track of entities for logging
                ent1 = ex1["entity"]
                ent2 = ex2["entity"]

            # prompt the LLM to compare the phrases (with example sentences) and choose the winner
            valid = False
            n_comp = 0
            while not valid and n_comp < 5:
                time.sleep(0.05)
                _, response, _ = llm_prompter.compare_sentences(sent1, sent2, self.llm_name)
                response = response.strip('"')
                if response == "sentence_1":
                    chosen_phrase = phrase1 
                    valid = True
                elif response == "sentence_2":
                    chosen_phrase = phrase2
                    valid = True
                n_comp += 1
            if n_comp == 5: 
                print("Error in LLM call! Inspect logs")
                chosen_phrase = "None"

            # log the result in JSON file
            self.log_result_jsonl(
                p1_phrase=phrase1,
                p2_phrase=phrase2,
                sent1_masked=sent1[0],
                sent2_masked=sent2[0],
                entity1=ent1,
                entity2=ent2,
                chosen_phrase=chosen_phrase
            )

    def run_all_comparisons(self, llm_prompter):
        for pair in self.pairs_to_compare:
            time.sleep(1)
            self.run_comparison(pair, llm_prompter)
    
    def log_result_jsonl(self, p1_phrase, p2_phrase, sent1_masked, sent2_masked, entity1, entity2, chosen_phrase):
        log_entry = {
            'phrase_1': p1_phrase,
            'phrase_2': p2_phrase,
            'sentence_1': sent1_masked,
            'sentence_2': sent2_masked,
            'entity_1': entity1,
            'entity_2': entity2,
            'chosen_phrase': chosen_phrase
        }
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')


if __name__ == "__main__":

    # load in vocabulary
    with open("../../data_resources/hedging_phrase_vocab.json", 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # choose number of repeated comparisons
    repeat = 10
    llm_name = "Gemini" # "GPT" or "Gemini" or "Claude" or "MedGemma"

    # set previous file (e.g. comparisons_claude.jsonl) if you want to rerun the same comparisons with the same sentences
    # set None if you want to start from scratch
    prev_comparison_file = "../../data_resources/hedging_phrase_comparisons/comparisons_gemini.jsonl"
            
    # run all comparisons. results are stored in json file
    result_path = f"../../data_resources/hedging_phrase_comparisons/comparisons_{llm_name}_NEW.jsonl"
    print(f"storing comparisons in {result_path}")
    comparer = UncertaintyComparer(llm_name, repeat, vocab, result_path, seed=42, sentence_pair_file=prev_comparison_file)

    # set up LLMs
    gemini_project="your-gemini-project-here"
    openai_endpoint="your-azure-openai-endpoint-here"
    openai_key="your-azure-openai-key-here"
    anthropic_key="your-anthropic-key-here"
    huggingface_token="your-huggingface-token-here"
    llm_prompter = LLMSentenceComparer(gemini_project, openai_endpoint, openai_key, anthropic_key, huggingface_token)

    comparer.run_all_comparisons(llm_prompter)