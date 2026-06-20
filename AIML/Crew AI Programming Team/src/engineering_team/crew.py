from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Code-writing agents need a large output budget so long modules / test files
# don't get truncated mid-file. Claude Sonnet 4.5 supports a big output window.
CODER_LLM = LLM(model="anthropic/claude-sonnet-4-5-20250929", max_tokens=16000)


@CrewBase
class EngineeringTeam():
    """EngineeringTeam crew: lead designs, backend implements, frontend builds a
    Gradio demo, and a test engineer writes unit tests."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ----- Agents -----

    @agent
    def engineering_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['engineering_lead'],
            verbose=True,
        )

    @agent
    def backend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_engineer'],
            llm=CODER_LLM,            # high max_tokens to avoid truncation
            verbose=True,
            # Code execution disabled (no Docker required). The agent still
            # writes the module; it just won't run it in a sandbox.
            max_retry_limit=3,
        )

    @agent
    def frontend_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['frontend_engineer'],
            verbose=True,
        )

    @agent
    def test_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['test_engineer'],
            llm=CODER_LLM,            # high max_tokens to avoid truncation
            verbose=True,
            # Code execution disabled (no Docker required).
            max_retry_limit=3,
        )

    # ----- Tasks -----

    @task
    def design_task(self) -> Task:
        return Task(config=self.tasks_config['design_task'])

    @task
    def code_task(self) -> Task:
        return Task(config=self.tasks_config['code_task'])

    @task
    def frontend_task(self) -> Task:
        return Task(config=self.tasks_config['frontend_task'])

    @task
    def test_task(self) -> Task:
        return Task(config=self.tasks_config['test_task'])

    # ----- Crew -----

    @crew
    def crew(self) -> Crew:
        """Creates the EngineeringTeam crew."""
        return Crew(
            agents=self.agents,    # created by the @agent decorators
            tasks=self.tasks,      # created by the @task decorators
            process=Process.sequential,
            verbose=True,
        )
