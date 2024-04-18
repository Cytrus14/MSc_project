import random
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer, MistralConfig
from RAGPipeline import RAGPipeline


class _ModelPipeline:
    def __init__(self):
        self.MODEL_END_TOKEN = '</s>'

        #self.rag_pipeline = RAGPipeline()
        
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", config=MistralConfig, device_map='cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)


    def gen_response(self, context, raw_prompt):
        #data = self.rag_pipeline.retrieve_relevant_data(raw_prompt)
        #print(data)
        
        tokenized_context = self.tokenizer(context, return_tensors="pt").to('cuda')
        generation_kwargs = dict(tokenized_context, streamer=self.streamer, do_sample=True, max_new_tokens=512)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        next(self.streamer)
        for el in self.streamer:
            el = el.replace(self.MODEL_END_TOKEN, '')
            yield(el)

ModelPipeline = _ModelPipeline()

