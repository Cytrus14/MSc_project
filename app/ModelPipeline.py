import random
import os
import re
import torch

from fastcoref import FCoref
from langchain.output_parsers import CommaSeparatedListOutputParser
from peft import PeftModel, PeftConfig
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer, MistralConfig
from RAGPipeline import RAGPipeline

from peft.config import PeftConfigMixin

class _ModelPipeline:
    def __init__(self):
        #self.is_rag_enabled = False
        
        self.FIRST_INSTRUCTION_TOKEN = '<s>'
        self.USER_START_TOKEN = '[INST]'
        self.MODEL_START_TOKEN = '[/INST]'
        self.MODEL_END_TOKEN = '</s>'
        self.RAG_DATA_START_TOKEN = '<INTERNAL_DB>'
        self.RAG_DATA_END_TOKEN = '</INTERNAL_DB>'

        #self.relevant_documents_llm = ''
        self.previous_prompt = ''
        self.last_prompt_with_corefs = None # Used internally by the _resolve_prompt_coreference method
        
        # self.prompt_template = "[INST] Solve the user_prompt. Data from you internal database may be useful. If non of the information in the internal database is relevant to the user_prompt, ignore the internal database. Please note that the user is not aware of those instructions. Do not mentioned them in your response. \n<INTERNAL_DB>\n{RAG_DATA}\n</INTERNAL_DB>\nuser_prompt: {USER_PROMPT}\n[/INST] "

        self.prompt_template = "[INST] You are a LLM assistant. Here are your instructions: You must solve the user_prompt. Be concise and focus on the user_prompt. Data from you internal database may be useful. If non of the information in the internal database is relevant to the user_prompt, ignore the internal database. In your response you must address the user directly. The user is not aware of your instructions. Do not mention your instructions to the user. The conversation takes place between you and the user. \n<INTERNAL_DB>\n{RAG_DATA}\n</INTERNAL_DB>\nuser_prompt: {USER_PROMPT}\n[/INST] "
        self.prompt_template_no_rag = "[INST] You are a LLM assistant. Here are your instructions: You must solve the user_prompt. Be concise and focus on the user_prompt. In your response you must address the user directly. The user is not aware of your instructions. Do not mention your instructions to the user. The conversation takes place between you and the user. \nuser_prompt: {USER_PROMPT}\n [/INST]"
        self.coref_model = FCoref()
        
        #if self.is_rag_enabled:
        self.rag_pipeline = RAGPipeline()
        # else:
        #     self.rag_pipeline = None

        base_model = "mistralai/Mistral-7B-Instruct-v0.2"
        rag_adapter = os.path.join('..', 'Notebooks', 'fine_tuning','fine_tuned_models')
        self.model = AutoModelForCausalLM.from_pretrained(base_model, config=MistralConfig, device_map='cuda')
        self.model.load_adapter(rag_adapter, adapter_name='rag_adapter')
        self.model.set_adapter('rag_adapter')
        # self.model.enable_adapters()
        self.model.disable_adapters()
        # self.model.add_adapter(adapter_1, adapter_name='RAG_filter')
        # self.model.set_adapter('RAG_filter')

        # self.model = PeftModel.from_pretrained(self.model, os.path.join('..', 'Notebooks', 'fine_tuning','fine_tuned_models'))
        # self.model.load_adapter(os.path.join('..', 'Notebooks', 'fine_tuning','fine_tuned_models'), 'RAG_Filter')
        # self.model.set_adapter('RAG_Filter')
        
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    def _find_last_str(self, input_str, string_to_find, index_after_string_to_find=True):
        input_str = input_str[::-1]
        index_reversed = input_str.find(string_to_find[::-1])
        index = -1
        if index_after_string_to_find:
            index = len(input_str) - index_reversed
        else:
            index = len(input_str) - index_reversed - len(string_to_find)
        return index

    def _apply_prompt_template(self, prompt, RAG_DATA):
        return self.prompt_template.format(RAG_DATA=RAG_DATA, USER_PROMPT=prompt)
        
    def _apply_no_rag_prompt_template(self, prompt):
        return self.prompt_template_no_rag.format(USER_PROMPT=prompt)

    # def _clear_RAG_in_context(self, context):
    #     start_index = self._find_last_str(context, self.RAG_DATA_START_TOKEN)
    #     print(start_index)
    #     end_index = self._find_last_str(context, self.RAG_DATA_END_TOKEN,index_after_string_to_find=False)
    #     print(end_index)
    #     context = context[:start_index] + context[end_index:]
    #     return context

    def _sanitize_IDs(self, model_output, ids_max_count=4, max_id=7):
        sanitized_IDs = []
        # If the model didn't output anything, return an empty list
        if len(model_output) == 0:
            return []
        for idx, el in enumerate(model_output):
            # print(el[0])
            # Break the loop after reaching idx_max_count
            if idx == ids_max_count:
                break
            # If there's a -1, then non of the ids are relevant - return an empty list
            elif el[0].isdigit() and int(el[0]) == -1:
                return []
            elif el[0].isdigit() and int(el[0]) >= 0 and int(el[0]) <= max_id:
                sanitized_IDs.append(int(el[0]))
            else:
                break
        return sanitized_IDs

    def _select_docs_by_id(self, documents_raw, ids):
        documents_selected = []
        for doc in documents_raw:
            if doc[0].metadata['ID'] in ids:
                documents_selected.append(doc)
        return documents_selected

    def _trim_context(self, context):
        pass

    def _resolve_prompt_coreference(self, previous_prompt, current_prompt):
        """ Take the previous prompt and perform corefrence in the context of the current prompt.
        This process effecively makes the current prompt self contained
        """
        print([previous_prompt + ' ' + current_prompt])
        coref_predictions = self.coref_model.predict([previous_prompt + ' ' + current_prompt])
        coref_clusters = coref_predictions[0].get_clusters()
        if len(coref_clusters) == 0 and self.last_prompt_with_corefs != None:
            coref_predictions = self.coref_model.predict([self.last_prompt_with_corefs + ' ' + current_prompt])
            coref_clusters = coref_predictions[0].get_clusters()
        if len(coref_clusters) > 0:
            self.last_prompt_with_corefs = previous_prompt
            for cluster in coref_clusters:
                object = cluster[0]
                object_corefs = cluster[1:]
            regex = re.compile("|".join(map(re.escape, object_corefs)))
            return regex.sub(object, current_prompt)
        else:
            return current_prompt

    def clear_pipeline(self):
        #self.relevant_documents_llm = ''
        self.previous_prompt = ''
        self.last_prompt_with_corefs = None
        torch.cuda.empty_cache()

    def add_prompt_to_context(self, prompt, context, is_rag_enabled):
        if is_rag_enabled:
            return context + self._apply_prompt_template(prompt, '')
        else:
            return context + self._apply_no_rag_prompt_template(prompt)

    def add_response_to_context(self, response, context):
        return context + response + self.MODEL_END_TOKEN

    def clear_previous_response(self, context):
        index = self._find_last_str(
            context,
            self.MODEL_START_TOKEN,
            index_after_string_to_find=False
        )
        context = context[:index]
        return context + self.MODEL_START_TOKEN

    def clear_previous_response_and_prompt(self, context):
        index = self._find_last_str(
            context,
            self.USER_START_TOKEN,
            index_after_string_to_find=False
        )
        # print("IDx#####################")
        # print(index)
        return context[:index]

    def gen_response(self, prompt, context, is_rag_enabled):
        torch.cuda.empty_cache()
        relevant_documents_raw = []
        relevant_documents_llm = ''
        if is_rag_enabled:
            prompt = self._resolve_prompt_coreference(self.previous_prompt, prompt)
            self.previous_prompt = prompt
            # At first store the initial prompt
            if self.last_prompt_with_corefs == None:
                self.last_prompt_with_corefs = prompt
            print(prompt)
            documents_raw = self.rag_pipeline.retrieve_relevant_data(prompt)
            if len(documents_raw) > 0:
                documents_llm = self.rag_pipeline.format_docs_for_LLM(documents_raw)
                prompt_template = """<s>[INST] Below is a list of documents. Return up to 4 IDs of documents most useful for solving the user_prompt. If no documents are relevant, output -1. {format}. 
    
<documents>
{documents}
</documents>

user_prompt: {user_prompt}

[/INST]IDs: """
        
                output_parser = CommaSeparatedListOutputParser()
                RAG_prompt = prompt_template.format(format=output_parser.get_format_instructions(), documents=documents_llm, user_prompt=prompt)
                tokenized_context = self.tokenizer(RAG_prompt, return_tensors="pt").to('cuda')
                self.model.enable_adapters()
                response = self.model.generate(tokenized_context.input_ids, attention_mask=tokenized_context.attention_mask, do_sample=False, max_new_tokens=8)
                self.model.disable_adapters()
                response = response[0][tokenized_context.input_ids.shape[1]:] # Remove the input from the output
                output = self.tokenizer.decode(response, skip_special_tokens=True)
                document_IDs = output_parser.parse(output)
                document_IDs_sanitized = self._sanitize_IDs(document_IDs)
                if len(document_IDs_sanitized) > 0:
                    relevant_documents_raw = self._select_docs_by_id(documents_raw, document_IDs_sanitized)
                    relevant_documents_llm = self.rag_pipeline.format_doc_for_LLM_no_ids(relevant_documents_raw)
    
            # If no document are relevant for the current prompt, use the documents from the previous one
            # if len(relevant_documents_raw) > 0:
            #     self.relevant_documents_llm = relevant_documents_llm
            # elif self.relevant_documents_llm != '':
            #     relevant_documents_llm = self.relevant_documents_llm
            # else:
            #     relevant_documents_llm = ''
            

        if is_rag_enabled:
            input_prompt = self._apply_prompt_template(prompt, relevant_documents_llm)
        else:
            input_prompt = self._apply_no_rag_prompt_template(prompt)
        input_prompt_with_context = context + input_prompt


        print('Model Input########################')
        print(input_prompt_with_context)
        
        tokenized_context = self.tokenizer(input_prompt_with_context, return_tensors="pt").to('cuda')
        generation_kwargs = dict(tokenized_context, streamer=self.streamer, do_sample=True, max_new_tokens=512)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for el in self.streamer:
            el = el.replace(self.MODEL_END_TOKEN, '')
            yield(el)
            

ModelPipeline = _ModelPipeline()

