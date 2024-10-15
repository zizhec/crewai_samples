from dotenv import load_dotenv
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# load dotenv:
load_dotenv()

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


random_integer_generator = Task(
    description="""Generate 4 random positive integers smaller than 12.""",
    agent = generic_assistent,
    expected_output="""A list of 4 integers""",
    # human_input=True
)
result_verifier_task = Task(
    description="""Verify the given solution is correct.""",
    agent = result_verifier,
    expected_output="""A boolean indicating whether the solution is correct""",
    # human_input=True
)
task_24 = Task(
    description="""Given 4 integers: {four_integers}, find a way to calculate 24 using the four basic arithmetic operations. 
    Provide the arithmetic expression used. The expression can use parentheses to indicate precedence. """,
    agent = generic_assistent,
    expected_output="""A complete formula.""",
    output_file='output/24.md',

    # human_input=True

)
crew = Crew(
    agents=[generic_assistent, result_verifier],
    tasks=[random_integer_generator, task_24, result_verifier_task],
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