import streamlit as st
from ModelPipeline import ModelPipeline
import random
import time

ai_avatar_path = './ai_avatar.png'
ai_name = 'Azi'

# Special Tokens used by the model
# Mistral 7B
FIRST_INSTRUCTION_TOKEN = '<s>'
USER_START_TOKEN = '[INST]'
MODEL_START_TOKEN = '[/INST]'
MODEL_END_TOKEN = '</s>'

# Style settings for better button arrangement
st.markdown(
    """
    <style>
        div[data-testid="column"] {
            width: fit-content !important;
            flex: unset;
        }
        div[data-testid="column"] * {
            width: fit-content !important;
        }
    """, unsafe_allow_html=True)

def gen_response_test(prompt):
    random_string = random.choice(["This is output 1", "This is output 2", "This is another output"])
    time.sleep(2)
    return prompt + random_string

# Find the index of the last occurrence of string_to_find in input_str
# index_after_string_to_find determines whether the returned index should
# be be after the sequence (True) or right before it (False)
def find_last_str(input_str, string_to_find, index_after_string_to_find=True):
    input_str = input_str[::-1]
    index_reversed = input_str.find(string_to_find[::-1])
    index = -1
    if index_after_string_to_find:
        index = len(input_str) - index_reversed
    else:
        index = len(input_str) - index_reversed - len(string_to_find)
    return index    

# def format_response(response):
#     index = find_last_str(
#         response,
#         '<end_of_turn>\n<start_of_turn>model')
#     return response[index:].strip()

def regenerate():
    placeholder.empty()
    st.session_state.messages.pop()
    index = find_last_str(
        st.session_state.context,
        MODEL_START_TOKEN,
        index_after_string_to_find=False
    )
    st.session_state.context = st.session_state.context[:index]
    print('Context after trimming##################')
    print(st.session_state.context)
    st.session_state.context += MODEL_START_TOKEN
    with placeholder2.container():
        with st.chat_message(ai_name, avatar=ai_avatar_path):
            response = st.write_stream(ModelPipeline.gen_response(st.session_state.context, st.session_state.raw_prompt))
    st.session_state.context += response + MODEL_END_TOKEN
    st.session_state.messages.append({"role": ai_name, "content": response})

def edit_prompt():
    placeholder.empty()
    st.session_state.in_edit_mode= True
    st.session_state.messages.pop()
    previous_user_message = st.session_state.messages.pop()
    index = find_last_str(
        st.session_state.context,
        USER_START_TOKEN,
        index_after_string_to_find=False
    )
    st.session_state.context = st.session_state.context[:index]

    chat_input_text = previous_user_message['content']
    js = f"""
        <script>
            function insertText() {{
                var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, "{chat_input_text}");
                var event = new Event('input', {{ bubbles: true}});
                chatInput.dispatchEvent(event);
                var iframe = parent.document.querySelector('iframe[data-testid="stIFrame"]');
                iframe.style.display='none';
            }}
            insertText({len(st.session_state.messages)});
        </script>
        """
    st.components.v1.html(js, height=0, width=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = FIRST_INSTRUCTION_TOKEN

if "raw_prompt" not in st.session_state:
    st.session_state.raw_prompt = ''

if "in_edit_mode" not in st.session_state:
    st.session_state.in_edit_mode = False

for idx, message in enumerate(st.session_state.messages):
    if message['role'] == ai_name:
        old_message = st.chat_message(message['role'], avatar=ai_avatar_path)
    else:
        old_message = st.chat_message(message['role'])
    old_message.markdown(message["content"])

if prompt := st.chat_input("Prompt"):
    st.session_state.context += USER_START_TOKEN
    message = st.chat_message("user")
    message.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt != None:    
    st.session_state.in_edit_mode = False
    st.session_state.raw_prompt = prompt
    st.session_state.context += prompt + MODEL_START_TOKEN
    with st.chat_message(ai_name, avatar=ai_avatar_path):
        response = st.write_stream(ModelPipeline.gen_response(st.session_state.context, st.session_state.raw_prompt))
    print(response)
    st.session_state.context += response + MODEL_END_TOKEN
    st.session_state.messages.append({"role": ai_name, "content": response})


_ = st.empty() # This workaround hides the buttons when generating a prompt
placeholder = st.empty()
placeholder2 = st.empty()
if len(st.session_state.messages) > 0 and not st.session_state.in_edit_mode:
    with placeholder.container():
        cl1, cl2, _, _, _, _ = st.columns(6)
        cl1.button("Regenerate", on_click=regenerate)
        cl2.button("Edit Prompt", on_click=edit_prompt)

print("Context###############")
print(st.session_state.context)
