import os
import time

import chromadb
import nltk
import spacy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from nltk.stem import WordNetLemmatizer

class RAGPipeline:
    def _search_vector_db(self, query, vector_db, k=512):
        query = 'search_query: ' + query
        most_similar_docs = vector_db.similarity_search_with_relevance_scores(query, k=k)
        return most_similar_docs
         
    def __init__(self, initial_doc_count=512):
        self.initial_doc_count = initial_doc_count
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
        embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={
                'device': 'cuda',
                'trust_remote_code': True
                }
            )
        chroma_client = chromadb.PersistentClient(path=os.path.join('..', 'Notebooks','chroma_data'))
        self.langchain_vector_db = Chroma(client=chroma_client, embedding_function=embedding_model)
        # Peform initial search to load everything into memory
        self._search_vector_db("Sample query", self.langchain_vector_db, k=1);

    def _extract_keywords(self, string):
        # Extract keywords from the prompt
        doc = self.nlp(string)
        keywords = set()
        for chunk in doc.noun_chunks:
            if not chunk.text.lower().strip() in nltk.corpus.stopwords.words('english'):
                text_doc = self.nlp(chunk.text)
                # Remove indirect articles and convert to lowercase
                text_words = [token.text for token in text_doc if not token.is_stop]
                text = ' '.join(text_words)
                # Keyword must be longer than 2 chars to be valid
                if len(text) > 2:
                    keywords.add(text.lower())
        # Convert keywords to their singular forms
        keywords = list(keywords)
        keywords_singular = [self.lemmatizer.lemmatize(word) for word in keywords]
        # print('Extracted keywords:')
        # print(keywords_singular)
        return keywords_singular
    
    def _contains_keywords_filter(self, keywords, docs):
        # Filter data by keywords
        filtered_data = []
        if len(keywords) > 0:
            for doc in docs:
                el = doc[0].page_content.lower()
                if any(keyword in el for keyword in keywords):
                    filtered_data.append(doc)
            return filtered_data
        else:
            return []

    def _add_id_to_doc_metadata(self, input_docs):
        output_docs = []
        for idx, doc in enumerate(input_docs):
            doc[0].metadata['ID'] = idx
            output_docs.append(doc)
        return output_docs
    
    def format_docs_for_LLM(self, docs):
        formated_documents = ""
        for doc in docs:
            page_content = "ID {}:\n".format(doc[0].metadata['ID'])
            page_content += "Title: {}\n".format(doc[0].metadata['title'])
            page_content += doc[0].page_content.replace("search_document: ", '', 1)
            page_content += "\n\n"
            formated_documents += page_content
        return formated_documents

    def format_doc_for_LLM_no_ids(self, docs):
        formated_documents = ""
        for doc in docs:
            page_content = "Title: {}\n".format(doc[0].metadata['title'])
            page_content += doc[0].page_content.replace("search_document: ", '', 1)
            page_content += "\n\n"
            formated_documents += page_content
        return formated_documents

    def retrieve_relevant_data(self, prompt):
        time_start = time.time()
        docs_with_score = self._search_vector_db(prompt, self.langchain_vector_db, self.initial_doc_count)
        time_end = time.time()
        print(f'Search vector DB time: {round(time_end - time_start, 5)}')
        
        time_start = time.time()
        keywords = self._extract_keywords(prompt)
        time_end = time.time()
        print(f'Extract keywords time: {round(time_end - time_start, 5)}')

        time_start = time.time()
        filtered_docs = self._contains_keywords_filter(keywords, docs_with_score)
        time_end = time.time()
        print(f'Filter keywords time: {round(time_end - time_start, 5)}')

        time_start = time.time()
        filtered_docs = filtered_docs[:8]
        time_end = time.time()
        print(f'Similarity filter time: {round(time_end - time_start, 5)}')
        
        # for doc in filtered_docs:
        #     print(doc[1])
            
        filtered_docs = self._add_id_to_doc_metadata(filtered_docs)
        # data = self._format_docs_for_LLM(filtered_docs)
        # return data
        return filtered_docs

