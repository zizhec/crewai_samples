from crewai import Agent, Task, Crew, Process
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from dotenv import load_dotenv
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

researcher = Agent(
    role='Researcher',
    goal='Uncover cutting-edge developments in {topic}',
    backstory="""You're a seasoned researcher with a knack for uncovering the latest developments in {topic}. Known for your ability to find the most relevant information and present it in a clear and concise manner.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    tools=[SerperDevTool()],
    LLM=ChatOpenAI(model="gpt-4o-mini")
)

reporter = Agent(
    role='Reporter',
    goal='Create detailed reports based on {topic} data analysis and research findings',
    backstory="""You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.""",
    verbose=True,
    # allow_delegation=True,
    cache=True,
    max_tokens=1000,
    LLM=ChatOpenAI(model="gpt-4o-mini")
)

research_task = Task(
    description="""Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2024.""",
    expected_output="""A list with 10 bullet points of the most relevant information about {topic}""",
    agent=researcher
)

reporting_task = Task(
    description="""Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.""",
    expected_output="""A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'""",
    agent=reporter,
    output_file='output/tech_invest.md'
)

crew = Crew(
    agents=[generic_assistent, researcher, reporter],
    tasks=[research_task, reporting_task],
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


inputs = {
    'topic': 'Role Based Agents'
}
result = crew.kickoff(inputs=inputs)


# Kick off the crew to start on it's tasks
# result = crew.kickoff()

print("######################")
print(result)
print(crew.usage_metrics)