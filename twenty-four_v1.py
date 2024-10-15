from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load dotenv:
load_dotenv()

# Initialize the OpenAI language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create agents
number_generator = Agent(role='Number Generator', goal='Generate 4 random integers between 1 and 9', backstory='You are responsible for creating the input for the 24 game', allow_delegation=False, llm=llm)

solver = Agent(role='24 Game Solver', goal='Solve the 24 game using the given numbers', backstory='You are an expert in arithmetic and problem-solving', allow_delegation=False, llm=llm)

checker = Agent(role='Solution Checker', goal='Verify if the proposed solution is correct', backstory='You are meticulous and can quickly verify arithmetic calculations', allow_delegation=False, llm=llm)

# Create tasks
generate_numbers_task = Task(
    description='Generate 4 random integers between 1 and 9',
    agent=number_generator,
    expected_output="Four random integers between 1 and 9, separated by spaces"
)

# Function to generate numbers
def generate_numbers():
    number_crew = Crew(agents=[number_generator], tasks=[generate_numbers_task], verbose=True)
    result = number_crew.kickoff()
    return result.strip()
    # return result[0].split(': ')[1].strip()

# Function to solve and check
def solve_and_check(numbers, solutions_memory):
    solve_task = Task(
        description=f"Solve the 24 game using these numbers: {numbers}. Your solution should use all four numbers exactly once, and use only addition, subtraction, multiplication, and division. The result must equal 24. Avoid these previously attempted solutions: {', '.join(solutions_memory)}",
        agent=solver,
        expected_output="A mathematical expression using the given numbers that equals 24"
    )

    check_task = Task(
        description=f"Check if the proposed solution for numbers {numbers} is correct. It should use all four numbers exactly once, use only basic arithmetic operations, and result in 24.",
        agent=checker,
        expected_output="Yes or No, followed by an explanation"
    )

    solve_check_crew = Crew(agents=[solver, checker], tasks=[solve_task, check_task], verbose=True)

    results = solve_check_crew.kickoff()
    return results[0], results[1]  # Return solution and check result

# Main loop
solutions_memory = set()
max_attempts = 10

# Generate numbers only once
print("Generating numbers...")
numbers = generate_numbers()
print(f"Numbers generated: {numbers}")

while max_attempts > 0:
    print("Solving...")
    solution, check_result = solve_and_check(numbers, solutions_memory)

    solution = solution.split(': ')[1].strip()
    is_correct = check_result.lower().startswith('yes')

    print(f"Proposed solution: {solution}")
    print(f"Is correct: {is_correct}")

    if is_correct:
        print("Correct solution found!")
        break
    else:
        solutions_memory.add(solution)
        print("Incorrect solution. Trying again...")

    max_attempts -= 1
    if max_attempts == 0:
        print("Max attempts reached. Exiting.")
        break

print("Process completed.")
