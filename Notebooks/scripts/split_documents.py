"""This module performs text splitting on multiple threads using the RecursiveCharacterTextSplitter from LangChain"""

import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

def text_splitter_wraper(documents_raw):
    huggingface_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192, rotary_scaling_factor=2)
    langchain_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer=huggingface_tokenizer, chunk_size=1536, chunk_overlap=50)
    return langchain_text_splitter.split_documents(documents_raw)
    
def main(documents_raw, thread_count=8):
    process_pool = multiprocessing.Pool(processes=thread_count)
    processed_documents = process_pool.map(text_splitter_wraper, documents_raw)

    process_pool.close()
    process_pool.join()
    return processed_documents

if __name__ == '__main__':
    main()

# import multiprocessing
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer

# def text_splitter_wraper(csv_input_file):
#     huggingface_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192, rotary_scaling_factor=2)
#     langchain_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer=huggingface_tokenizer, chunk_size=1536, chunk_overlap=50)
#     return langchain_text_splitter.split_documents(documents_raw)
    
# def main(csv_input_file, thread_count=8):
#     #Prepare the csv loader
#     langchain_loader = CSVLoader(
#         file_path=csv_input_file,
#         metadata_columns=['title', 'categories']
#     )

#     # Split the documents using multiple threads and an external splitting script
#     documents_raw = langchain_loader.load()
#     documents_raw_sublists = split_list(documents_raw, thread_count)
    
#     process_pool = multiprocessing.Pool(processes=thread_count)
#     documents_split_sublists = process_pool.map(text_splitter_wraper, documents_raw_sublists)

#     process_pool.close()
#     process_pool.join()

#     documents_split = list(itertools.chain(*documents_split_sublists))
#     return documents_split

# if __name__ == '__main__':
#     main()