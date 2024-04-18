import os

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
        keywords = []
        for chunk in doc.noun_chunks:
            if not chunk.text.lower().strip() in nltk.corpus.stopwords.words('english'):
                text = chunk.text
                # Remove indirect articles
                text = text.replace('a ', '').replace('an ', '').strip()
                keywords.append(text)
        # Convert keywords to their singular forms
        keywords_singular = [self.lemmatizer.lemmatize(word) for word in keywords]
        return keywords_singular
    
    def _contains_keywords_filter(self, keywords, docs):
        # Filter data by keywords
        filtered_data = []
        if len(keywords) > 0:
            for doc in docs:
                el = doc[0].page_content
                if any(keyword in el for keyword in keywords):
                    filtered_data.append(doc)
            return filtered_data
        else:
            return docs
    
    def _format_docs_for_LLM(self, docs):
        formated_documents = ""
        for idx, doc in enumerate(docs):
            page_content = "ID {}:\n".format(idx)
            page_content += "Title: {}\n".format(doc[0].metadata['title'])
            page_content += doc[0].page_content.replace("search_document: ", '', 1)
            page_content += "\n\n"
            formated_documents += page_content
        return formated_documents

    def retrieve_relevant_data(self, prompt):
        docs_with_score = self._search_vector_db(prompt, self.langchain_vector_db, self.initial_doc_count)
        keywords = self._extract_keywords(prompt)
        filtered_docs = self._contains_keywords_filter(keywords, docs_with_score)
        filtered_docs = filtered_docs[:8]
        data = self._format_docs_for_LLM(filtered_docs)
        return data

