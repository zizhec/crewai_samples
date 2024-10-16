from dotenv import load_dotenv
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# load dotenv:
load_dotenv()

math_problem_composer = Agent(
    role='Math Problem Composer',
    goal='Compose a math problem',
    backstory="""As a math problem composer, you can compose a difficult math problem.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    LLM =ChatOpenAI(model="gpt-4o-mini")
    # (optional) llm=another_llm
)

generic_assistent = Agent(
    role='Generic Assistent',
    goal='Assist with any task',
    backstory="""As a helpful assistant, you can assist with any task.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    LLM=ChatOpenAI(model="gpt-4o-mini")
    # (optional) llm=another_llm
)

math_task = Task(
    description="""Propose a difficult math problem.""",
    agent = math_problem_composer,
    expected_output="""A math problem""",
    # human_input=True
)
problem_solver = Agent(
    role='Math Solution Proposer',
    goal='Find one solution for the problem',
    backstory="""As a senior math expert, you can find a very promising solution for the given math problem.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    LLM=ChatOpenAI(model="gpt-4o-mini")
    # (optional) llm=another_llm
)
result_verifier = Agent(
    role='Result Verifier',
    goal='Verify the solution is correct',
    backstory="""As a result verifier, you can verify whether the solution is correct.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    LLM=ChatOpenAI(model="gpt-4o-mini")
    # (optional) llm=another_llm
)
math_problem_solver = Task(
    description="""Find a solution for the given math problem.""",
    agent = problem_solver,
    expected_output="""A solution for the math problem""",
    # human_input=True
)
result_verifier_task = Task(
    description="""Verify the given solution is correct.""",
    agent = result_verifier,
    expected_output="""A boolean indicating whether the solution is correct""",
    # human_input=True
)
final_result_summarizer = Agent(
    role='Final Result Summarizer',
    goal='Summarize the final result',
    backstory="""As a final result summarizer, you can summarize the final result.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens = 1000,
    LLM=ChatOpenAI(model="gpt-4o-mini")
    # (optional) llm=another_llm
)

crew = Crew(
    agents=[math_problem_composer, generic_assistent, result_verifier, final_result_summarizer],
    tasks=[math_task, math_problem_solver, result_verifier_task, ],
    manager_llm=ChatOpenAI(model="gpt-4o-mini"),
    process=Process.sequential,
    verbose=True,
    memory=True,
    cache=True,
    language='Chinese',
    # planning=True,
    # planning_llm=ChatOpenAI(model="gpt-4o-mini"),
    # usage_metrics=True
)

result = crew.kickoff()

print("######################")
print(result)
print(crew.usage_metrics)