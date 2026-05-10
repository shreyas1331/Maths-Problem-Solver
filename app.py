import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Groq")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# 1. Main tool-calling LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# 2. Strict math engine
llm_math_engine = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for searching the Internet to find general information on non-mathematical topics."
)

math_chain = LLMMathChain.from_llm(llm=llm_math_engine)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for evaluating explicit mathematical or arithmetic calculations expressions (e.g. 8 - 2)."
)

# Structured Reasoning Prompt to prevent tool confusion loops
prompt_text = """
You are a reasoning model. Break down the logic step-by-step to answer the query.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt_text)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning_Tool",
    func=chain.run,
    description="Useful for breaking down word problems, logical riddles, and multi-step text math questions before calculating."
)

tools = [wikipedia_tool, calculator, reasoning_tool]

# 3. Modern Tool Calling Prompt Design
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant equipped with specialized tools. Always use the most specific tool available to solve the user's problem. Once you obtain the final answer from a tool, provide it directly to the user."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Initialize Modern Native Tool Calling Agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)
assistant_agent = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5 # Safety fallback boundary
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Interaction Input UI
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
            
            # Execute with modern agent format
            response_obj = assistant_agent.invoke({"input": question}, callbacks=[st_cb])
            response = response_obj["output"]
            
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter a question")
