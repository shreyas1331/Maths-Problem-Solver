import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
# FIX: Use the updated community package to avoid deployment deprecation crashes
from langchain_community.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Groq")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# 1. Main conversational LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# 2. Strict math engine with 0 temperature to stop conversational formatting
llm_math_engine = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned."
)

math_chain = LLMMathChain.from_llm(llm=llm_math_engine)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only raw mathematical expressions should be passed to this tool (e.g., (5-2)+(7-3))."
)

prompt = """
You are an expert agent tasked with solving a user's question. 
Break down the problem logically, show your steps clearly, use tools if necessary, and provide a clear final answer.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# 3. Dedicated tool binding configuration
# Automatically routes complex queries into a structured tool pipeline natively
tools_map = {
    "wikipedia": wikipedia_tool,
    "calculator": calculator
}

llm_with_tools = llm.bind_tools([wikipedia_tool, calculator])

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Let's start the interaction
question = st.text_area(
    "Enter your question:",
    value="If I have 8 bananas and 2 I have given to someone then how much are left with me"
)

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # 4. Native Engine Invocation
            # Queries the model directly with tools to execute calculations with zero parsing loops
            ai_msg = llm_with_tools.invoke(question, callbacks=[st_cb])
            
            # If the model requests a tool call, handle it natively
            if ai_msg.tool_calls:
                tool_call = ai_msg.tool_calls[0]
                tool_name = tool_call["name"].lower()
                tool_args = tool_call["args"]
                
                # Extract clean string arguments for execution
                query_arg = list(tool_args.values())[0] if tool_args else question
                
                if "calc" in tool_name or "math" in tool_name:
                    result = calculator.run(query_arg)
                else:
                    result = wikipedia_tool.run(query_arg)
                
                # Combine original reasoning with tool output for final synthesis
                final_prompt = f"Based on the tool result: '{result}', solve the user question: '{question}'"
                response = llm.invoke(final_prompt).content
            else:
                response = ai_msg.content
            
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter a question")
