from crewai import Agent, Task, Crew
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tabulate import tabulate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# search_tool = SerperDevTool()
search_tool = DuckDuckGoSearchRun()

tech_researcher = Agent(
    role='Technology Researcher',
    goal='Uncover cutting-edge developments in {tech_name}',
    backstory='You are an expert in technology trends and solutions. You use web search to find comprehensive information about various technologies and their implementations.',
    tools=[search_tool],
    verbose=True,
    llm=llm
)

# researcher = Agent(
#     role='Researcher',
#     goal='Uncover cutting-edge developments in {topic}',
#     backstory="""You're a seasoned researcher with a knack for uncovering the latest developments in {topic}. Known for your ability to find the most relevant information and present it in a clear and concise manner.""",
#     verbose=True,
#     # allow_delegation=True,
#     cache=True,
#     max_tokens=1000,
#     tools=[SerperDevTool()],
#     LLM=ChatOpenAI(model="gpt-4o-mini")
# )

solution_analyst = Agent(
    role='Solution Analyst',
    goal='Analyze solutions, their strengths, use-cases, and funding',
    backstory='You specialize in analyzing tech solutions, their applications, and financial backgrounds.',
    tools=[search_tool],
    verbose=True,
    llm=llm
)

report_writer = Agent(
    role='Report Writer',
    goal='Compile a comprehensive table report on the technology solutions',
    backstory='You are a technical writer skilled in creating clear, concise, and well-structured reports on complex topics.',
    verbose=True,
    llm=llm
)

task1 = Task(
    description='''Research and find the top 5 most influential solutions for {tech_name}.
     For each solution, provide:
     1. Solution name
     2. Developer or company behind the solution
     3. Brief description of the solution''',
    agent=tech_researcher,
    expected_output="A list of 5 influential solutions with their names, developers/companies, and brief descriptions"
)

task2 = Task(
    description='''For each solution found in task 1, research and provide:
     1. Main strength of the solution
     2. Primary use-cases
     3. Two main pros
     4. Two main cons
     5. Whether the company/developer has raised any funding (Yes/No, and if possible, how much)''',
    agent=solution_analyst,
    expected_output="Detailed information about each solution, including strengths, use-cases, pros, cons, and funding status"
)

task3 = Task(
    description='''Create a table report with the following columns:
     Solution Name | Developer/Company | Solution Strength | Primary Use-cases | Pros | Cons | Funding Raised
     Use the information from previous tasks to fill in the table.
     Format the table using the tabulate library with headers.
     Ensure all information is concise and fits well in a table format.''',
    agent=report_writer,
    expected_output="A formatted table containing all the gathered information about the solutions"
)

crew = Crew(
    agents=[tech_researcher, solution_analyst, report_writer],
    tasks=[task1, task2, task3],
    verbose=True,
    memory=True,
    cache=True,
    manager_llm=llm,
)

inputs = {
    'tech_name': 'Agent Orchestration'
}
result = crew.kickoff(inputs=inputs)
print(crew.usage_metrics)
print(result)