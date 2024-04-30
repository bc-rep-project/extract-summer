# import unittest
from langchain.llms import OpenAI, BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor, ReActAgent
from langchain.tools import SerpAPIWrapper
from langchain.callbacks.manager import CallbackManager


class LangChainLLMWrapper:
    def __init__(self, llm: BaseLLM, model_name: str = "gpt-3.5-turbo", memory_type: str = "buffer"):
        self.llm = llm
        self.model_name = model_name

        memory_options = {
            "buffer": ConversationBufferMemory,
            "summary": ConversationSummaryMemory,
        }
        self.memory = memory_options[memory_type](memory_key="chat_history")

    def generate_text(
        self, prompt: str, chain_of_thought: list[str] = None, callbacks: list = None
    ) -> str:
        if chain_of_thought:
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["input", "chain_of_thought"],
                    template="{chain_of_thought}\n\nHuman: {input}\nAssistant:",
                ),
                memory=self.memory,
                callback_manager=CallbackManager(callbacks),
            )
            return llm_chain.run({"input": prompt, "chain_of_thought": chain_of_thought})
        else:
            return self.llm(prompt)

    def use_tools(
        self,
        objective: str,
        tools: list[Tool],
        agent_type: str = "zero-shot",
        callbacks: list = None,
    ) -> str:
        agent_options = {
            "zero-shot": ZeroShotAgent,
            "react": ReActAgent,
        }
        agent = agent_options[agent_type](llm_chain=self.llm, allowed_tools=tools)
        tool_names = [tool.name for tool in tools]
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            callback_manager=CallbackManager(callbacks),
        )
        return executor.run(objective)


# Example usage with tools and different agent and memory:
tools = [
    Tool(
        name="Search",
        func=SerpAPIWrapper().run,
        description="useful for when you need to answer questions about current events",
    )
]
wrapper = LangChainLLMWrapper(
    llm=OpenAI(temperature=0.9, openai_api_key="YOUR_API_KEY"), memory_type="summary"
)
output = wrapper.use_tools(
    "What was the score of the most recent Superbowl?", tools=tools, agent_type="react"
)
print(output)

# class ResearchAgent:
#     def __init__(self, llm_wrapper: LangChainLLMWrapper):
#         self.llm_wrapper = llm_wrapper
#         self.tools = [
#             Tool(
#                 name="Search",
#                 func=SerpAPIWrapper().run,
#                 description="useful for when you need to answer questions about current events or find information",
#             )
#         ]

#     def research(self, query: str) -> str:
#         return self.llm_wrapper.use_tools(objective=query, tools=self.tools)


# class TestingAgent:
#     def __init__(self, llm_wrapper: LangChainLLMWrapper):
#         self.llm_wrapper = llm_wrapper
#         self.llm = llm_wrapper.llm

#     def generate_test_cases(self, code: str, function_name: str) -> list[str]:
#         prompt = PromptTemplate(
#             input_variables=["code", "function_name"],
#             template="Given the following code:\n```python\n{code}\n```\nGenerate a list of test cases for the function `{function_name}`.",
#         )
#         test_cases_str = self.llm_wrapper.generate_text(
#             prompt.format(code=code, function_name=function_name)
#         )
#         return test_cases_str.strip().split("\n")

#     def execute_tests(self, test_cases: list[str]) -> str:
#         class Test(unittest.TestCase):
#             pass

#         for i, test_case in enumerate(test_cases):
#             def test_func(self, test_case=test_case):
#                 exec(test_case)

#             setattr(Test, f'test_case_{i}', test_func)

#         suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
#         runner = unittest.TextTestRunner()
#         test_results = runner.run(suite)
#         return str(test_results)