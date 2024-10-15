import os

from crewai import Agent, Task, Crew
from crewai_tools import GithubSearchTool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool

from tabulate import tabulate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
proxies = {
    "http": os.environ["HTTP_PROXY"],
    "https": os.environ["HTTP_PROXY"],
}

llm = ChatOpenAI(model="gpt-4o-mini")

search_tool = SerperDevTool()

github_tool = GithubSearchTool(
    gh_token=os.environ["GITHUB_TOKEN"],
	content_types=['repo'] # Options: code, repo, pr, issue
)

github_researcher = Agent(role='GitHub Researcher', goal='Find influential GitHub repositories for the given technology', backstory='You are an expert in GitHub trends and open-source projects.', tools=[
    github_tool, search_tool], verbose=True, llm=llm)

solution_analyst = Agent(role='Solution Analyst', goal='Analyze solutions, their companies, and funding', backstory='You specialize in analyzing tech solutions, companies, and their financial backgrounds.', tools=[
    search_tool], verbose=True, llm=llm)

report_writer = Agent(role='Report Writer', goal='Compile a comprehensive table report on the technology', backstory='You are a technical writer skilled in creating clear, concise, and well-structured reports on complex topics.', verbose=True, llm=llm)

# Define tasks
task1 = Task(
    description='''Find the top 5 most starred GitHub repositories for {tech_name}.
     For each repository, provide:
     1. Repository name
     2. Number of stars
     3. Brief description of the solution''',
    agent=github_researcher,
    expected_output="A list of 5 GitHub repositories with their names, star counts, and brief descriptions"
)

task2 = Task(
    description='''For each solution found in task 1, research and provide:
     1. Company name behind the solution
     2. Whether the company has raised any funding (Yes/No)
     3. Main strength of the solution
     4. Primary use-case
     5. Two main pros
     6. Two main cons''',
    agent=solution_analyst,
    expected_output="Detailed information about each solution, including company details, funding status, strengths, use-cases, pros, and cons"
)

task3 = Task(
    description='''Create a table report with the following columns:
     Solution Name | GitHub Stars | Strength | Use-case | Pros | Cons | Company Name | Funding Raised
     Use the information from previous tasks to fill in the table.
     Format the table using the tabulate library with headers.
     Ensure all information is concise and fits well in a table format.''',
    agent=report_writer,
    expected_output="A formatted table containing all the gathered information about the solutions"
)

# Create crew
crew = Crew(
    agents=[github_researcher, solution_analyst, report_writer],
    tasks=[task1, task2, task3],
    verbose=True,
    memory=True,
    cache=True,
    manager_llm=llm,
)

# Start the process
inputs = {
    'tech_name': 'Role Based Agents'
}
result = crew.kickoff(inputs=inputs)
print(crew.usage_metrics)
print(result)
