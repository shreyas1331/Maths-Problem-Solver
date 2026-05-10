import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain_community.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Groq")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# 1. Core LLM Clients
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
llm_math_engine = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)

## Initializing the Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="For searching general information on historical, geographical, or non-mathematical topics."
)

custom_math_prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate to a pure Python numeric expression for numexpr. No text.\nQuestion: {question}\n```python\n"
)
math_chain = LLMMathChain.from_llm(llm=llm_math_engine, prompt=custom_math_prompt)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="For executing raw arithmetic math formulas or expressions (e.g. 8-2, 12*25)."
)

# 2. Strict Router Prompt Configuration
router_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an routing agent. Analyze the question and select the single best tool to use.\n"
        "Available tools:\n"
        "- 'Calculator': If the question requires executing explicit math formulas, arithmetic equations, or numeric calculations.\n"
        "- 'Wikipedia': If the question requires looking up factual real-world definitions or historical events.\n"
        "- 'Direct': If the question can be answered with simple logic or direct context without tools.\n\n"
        "Return ONLY a clean valid JSON object with keys 'tool' and 'input'. Do not include markdown code blocks or explanations.\n"
        "Example output: {{\n  \"tool\": \"Calculator\",\n  \"input\": \"8 - 2\"\n}}\n\n"
        "Question: {question}"
    )
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
        with st.spinner("Processing..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Route calculation logic natively
            router_input = router_prompt.format(question=question)
            router_response = llm.invoke(router_input).content.strip()
            
            # Clean up potential markdown blocks added by chat models
            if router_response.startswith("```"):
                router_response = router_response.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                if router_response.startswith("json"):
                    router_response = router_response[4:].strip()

            try:
                decision = json.loads(router_response)
                selected_tool = decision.get("tool", "Direct")
                tool_input = decision.get("input", question)
            except Exception:
                selected_tool = "Direct"
                tool_input = question

            # 3. Native Tool Execution Block (Bypasses problematic agent frameworks)
            if selected_tool == "Calculator":
                st.info("🔄 Invoking Calculator Tool...")
                try:
                    tool_output = calculator.run(tool_input)
                except Exception as e:
                    tool_output = f"Calculation issue: {str(e)}"
            elif selected_tool == "Wikipedia":
                st.info("🔍 Invoking Wikipedia Search Tool...")
                tool_output = wikipedia_tool.run(tool_input)
            else:
                tool_output = "No tool needed."

            # Synthesis prompt to generate final formatted output
            synthesis_prompt = (
                f"Answer the user's question clearly and step-by-step.\n"
                f"User Question: {question}\n"
                f"Tool Used: {selected_tool}\n"
                f"Tool Execution Output Context: {tool_output}\n\n"
                f"Provide a point-wise logical explanation followed by the final answer."
            )
            
            final_response = llm.invoke(synthesis_prompt).content
            
            st.session_state.messages.append({'role': 'assistant', "content": final_response})
            st.write('### Response:')
            st.success(final_response)
    else:
        st.warning("Please enter a question")
