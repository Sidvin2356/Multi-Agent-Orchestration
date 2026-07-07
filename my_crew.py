import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


load_dotenv()

# Configuration Block
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if OPENAI_KEY:
    llm_brain = LLM(provider="openai", model="gpt-3.5-turbo", api_key=OPENAI_KEY)
elif GEMINI_KEY:
    # since gemini is available for free, i've used it here
    llm_brain = LLM(model="gemini-2.5-flash", api_key=GEMINI_KEY, use_native=True)
else:
    raise RuntimeError("Set OPENAI_API_KEY or GEMINI_API_KEY in the environment before running")

#Defining the Agents
developer = Agent(
    role='Senior Robotics Software Engineer',
    goal='Write clean, efficient Python code for robotic simulation and control.',
    backstory='You are an expert in writing ROS 2 nodes and Python scripts for robotic arms. You write highly modular and well-documented code.',
    llm=llm_brain,
    verbose=True, # To see the Ai Agent's thought process in the console    
    allow_delegation=False # Forces the agent to do the work itself
)

reviewer = Agent(
    role='Code QA Specialist',
    goal='Ensure the code is safe, bug-free, and adheres to best practices.',
    backstory='You are a meticulous reviewer who catches edge cases, infinite loops, and missing imports in Python scripts.',
    llm=llm_brain,
    verbose=True,
    allow_delegation=False
)

#Defining the tasks here
write_code = Task(
    description='Write a simple Python script for a robotic arm to execute a pick-and-place sequence using basic joint control logic.',
    expected_output='A clean, commented Python script.',
    agent=developer
)

review_code = Task(
    description='Review the provided Python script for the pick-and-place sequence. Check for logical errors, missing imports, and add a brief explanation of how to improve it.',
    expected_output='The final revised Python script, followed by a bulleted list of review notes.',
    agent=reviewer
)

#Building the Orchestration Crew
crew = Crew(
    agents=[developer, reviewer],
    tasks=[write_code, review_code],
    process=Process.sequential # This forces Task 1 to finish before Task 2 starts
)

#start crew.ai
print("Starting the AI Crew...")
result = crew.kickoff()

print("\n==================================")
print("FINAL RESULT:")
print("==================================")
print(result)