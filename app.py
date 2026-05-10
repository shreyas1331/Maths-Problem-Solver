import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Groq")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# 1. Initialize models with zero-temperature fallback to enforce deterministic tool routing
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for searching the Internet to find general information on non-mathematical topics."
)

# 2. Fix the LLMMathChain output format conflict natively using an explicit internal prompt override
custom_math_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "Translate this math problem into a clean Python expression that runs with the numexpr library. "
        "Do not write text, introductions, markdown or explanations. Return ONLY the code format required.\n"
        "Question: {question}\n"
        "```python\n"
    )
)

math_chain = LLMMathChain.from_llm(llm=llm, prompt=custom_math_prompt)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for evaluating arithmetic, numeric or equations expressions. Pass only raw expressions (e.g. 8-2)."
)

# 3. Create a logic block to isolate word problems from simple tool execution loops
prompt_text = "Solve this multi-step logic text reasoning query clearly step-by-step: {question}"
prompt_template = PromptTemplate(input_variables=["question"], template=prompt_text)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning_Tool",
    func=chain.run,
    description="Useful for breaking down raw textual word math problems, counting scenarios, or logic questions."
)

tools = [wikipedia_tool, calculator, reasoning_tool]

# 4. Use STRUCTURED_CHAT agent type to prevent standard text loop parsing errors in LangChain 0.1.14
assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # Strict iteration fallback boundary
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
            
            # Execute safely inside the native LangChain 0.1.14 agent architecture
            response = assistant_agent.run(question, callbacks=[st_cb])
            
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter a question")
