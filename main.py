import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def write_output_files(result):
    all_output = str(result)
    print(f"Length of all_output: {len(all_output)}")
    print("All output content:")
    print(all_output)
    
    with open('output.md', "w") as file:
        file.write("# Machine Learning Project Summary\n\n")
        file.write(all_output)

        # Python Library Dependencies
        file.write("\n\n## Python Library Dependencies\n\n")
        libraries = [
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "scipy",
            "category_encoders"
        ]
        for lib in libraries:
            file.write(f"- {lib}\n")

    print("Complete output has been exported to output.md")
    
    # Print the content of the file after writing
    with open('output.md', 'r') as file:
        print(f"Content of output.md:\n{file.read()}")

def main():
    """
    Main function to initialize and run the CrewAI Machine Learning Assistant.

    This function sets up a machine learning assistant using the Llama 3 model with the ChatGroq API.
    It provides a text-based interface for users to define, assess, and solve machine learning problems
    by interacting with multiple specialized AI agents. The function outputs the results to the console
    and writes them to a markdown file.

    Steps:
    1. Initialize the ChatGroq API with the specified model and API key.
    2. Display introductory text about the CrewAI Machine Learning Assistant.
    3. Create and configure four AI agents:
        - Problem_Definition_Agent: Clarifies the machine learning problem the user wants to solve, 
            identifying the type of problem (e.g., classification, regression) and any specific requirements.
        - Data_Assessment_Agent: Thoroughly evaluates the provided data, assesses its quality, detects and handles data issues, 
            and suggests comprehensive preprocessing steps to prepare the data for machine learning models.
        - Model_Recommendation_Agent: Suggests suitable machine learning models based on the problem definition 
            and data assessment, providing reasons for each recommendation.
        - Starter_Code_Generator_Agent: Generates starter Python code for the project, including data loading, cleaning, 
            preprocessing, model definition, training, cross-validation (if recommended), and model 
            comparison visualizations, based on findings from the problem definitions, data assessment, 
            specific model recommendations, cross-validation assessment, and visualization recommendations.
    4. Prompt the user to describe their machine learning problem.
    5. Check if a .csv file is available in the current directory and try to read it as a DataFrame.
    6. Define tasks for the agents based on user input and data availability.
    7. Create a Crew instance with the agents and tasks, and run the tasks.
    8. Print the results and write them to an output markdown file.
    """

    # model = 'llama3-8b-8192'
    model = "llama3-70b-8192"

    llm = ChatGroq(
        temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name=model
    )

    print("CrewAI Machine Learning Assistant")
    multiline_text = """
    The CrewAI Machine Learning Assistant is designed to guide users through the process of defining, assessing, and solving machine learning problems. It leverages a team of AI agents, each with a specific role, to clarify the problem, evaluate the data, recommend suitable models, and generate starter Python code. Whether you're a seasoned data scientist or a beginner, this application provides valuable insights and a head start in your machine learning projects.
    """

    print(multiline_text)

    Problem_Definition_Agent = Agent(
        role="Problem_Definition_Agent",
        goal="Define the machine learning problem clearly and concisely.",
        backstory="You are an expert in understanding and defining machine learning problems.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Check if there is a .csv file in the current directory
    csv_files = [file for file in os.listdir() if file.endswith(".csv")]
    if csv_files:
        sample_fp = csv_files[0]
        try:
            # Attempt to read the uploaded file as a DataFrame
            df = pd.read_csv(sample_fp)
            data_info = f"""
            Dataset Information:
            - Filename: {sample_fp}
            - Number of rows: {df.shape[0]}
            - Number of columns: {df.shape[1]}
            - Columns: {', '.join(df.columns)}
            - Data types:
            {df.dtypes.to_string()}
            
            First 5 rows of the dataset:
            {df.head().to_string()}
            """
            print("Data successfully loaded:")
            print(data_info)
        except Exception as e:
            print(f"Error reading the file: {e}")
            data_info = "No valid CSV file found or error reading the file."
    else:
        data_info = "No CSV file found in the current directory."

    Data_Assessment_Agent = Agent(
        role="Data_Assessment_Agent",
        goal="Assess and preprocess the data for the AI problem.",
        backstory=f"You are a data scientist specializing in data assessment, exploration, and preprocessing. Here's the data you're working with:\n{data_info}",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    AI_Technique_Recommendation_Agent = Agent(
        role="AI_Technique_Recommendation_Agent",
        goal="Recommend suitable AI techniques, including machine learning, deep learning, and other approaches.",
        backstory=f"You are an expert in various AI techniques and their applications. Here's the data you're working with:\n{data_info}",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Code_Generator_Agent = Agent(
        role="Code_Generator_Agent",
        goal="Generate comprehensive Python code for the entire AI pipeline.",
        backstory="You are a skilled AI engineer proficient in writing clean, efficient code for various AI techniques.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Cross_Validation_Agent = Agent(
        role="Cross_Validation_Agent",
        goal="Determine if k-fold cross-validation is appropriate for the given problem and dataset, and implement it when suitable.",
        backstory="You are an expert in model validation techniques. Your task is to assess whether k-fold cross-validation is necessary based on the problem type, dataset size, and other relevant factors. If appropriate, you implement and explain  the cross-validation process.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Visualization_Agent = Agent(
        role='Visualization_Agent',
        goal="Generate visualizations comparing the performance of the recommended models using matplotlib and seaborn.",
        backstory="You are a data visualization expert specializing in machine learning model comparisons. Your task is to create clear, informative visualizations that help users understand and compare the performance of the specific machine learning models recommended for their problem. You use matplotlib and seaborn to create appropriate visualizations based on the problem type and the recommended models.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # user_question = input("Describe your ML problem: ")
    user_question = "Develop a model to predict the price of houses using the given data"
    data_upload = False
    # Check if there is a .csv file in the current directory
    if any(file.endswith(".csv") for file in os.listdir()):
        sample_fp = [file for file in os.listdir() if file.endswith(".csv")][0]
        try:
            # Attempt to read the uploaded file as a DataFrame
            df = pd.read_csv(sample_fp).head(5)

            # If successful, set 'data_upload' to True
            data_upload = True

            # Display the DataFrame in the app
            print("Data successfully uploaded and read as DataFrame:")
            print(df)
        except Exception as e:
            print(f"Error reading the file: {e}")

    if user_question:

        task_define_problem = Task(
            description="""Define the machine learning problem based on the user's input. Include:
            1. Clear problem statement
            2. Type of machine learning problem (e.g., classification, regression)
            3. Specific requirements or constraints
            4. Potential challenges""",
            agent=Problem_Definition_Agent,
            expected_output="A comprehensive problem definition with the elements listed above.",
        )

        task_assess_data = Task(
            description=f"""Assess the data and suggest preprocessing steps. Include:
            1. Data collection and exploration insights
            2. Data quality assessment
            3. Necessary preprocessing steps
            4. Feature engineering suggestions
            Provide code snippets for data loading, exploration, and preprocessing.
            
            Data Information:
            {data_info}""",
            agent=Data_Assessment_Agent,
            expected_output="A detailed data assessment report with code snippets for preprocessing and feature engineering.",
        )

        task_recommend_technique = Task(
            description=f"""Recommend suitable AI techniques. Include:
            1. List of recommended techniques (machine learning, deep learning, and other AI approaches) with rationale
            2. Pros and cons of each technique
            3. Suggestions for technique selection criteria
            4. Any ensemble or hybrid methods to consider
            
            Base your recommendations on this data:
            {data_info}""",
            agent=AI_Technique_Recommendation_Agent,
            expected_output="A comprehensive list of recommended AI techniques with detailed explanations.",
        )

        task_generate_code = Task(
            description="""Generate Python code for the entire AI pipeline. Include:
            1. Data loading and preprocessing
            2. Feature engineering (if applicable)
            3. Model/technique implementation (for all recommended approaches)
            4. Training and evaluation
            5. Cross-validation or other validation methods (if applicable)
            6. Hyperparameter tuning suggestions
            7. Performance comparison visualizations
            Ensure the code is well-commented and follows best practices.""",
            agent=Code_Generator_Agent,
            expected_output="Complete, well-structured Python code for the entire AI pipeline.",
        )

        task_cross_validation = Task(
            description="""Implement k-fold cross-validation for the recommended models. Include:
            1. Explanation of cross-validation importance
            2. Code for implementing k-fold cross-validation
            3. Guidelines for interpreting cross-validation results""",
            agent=Cross_Validation_Agent,
            expected_output="Detailed explanation and code for k-fold cross-validation implementation.",
        )

        task_visualize_comparison = Task(
            description="""Create visualizations to compare the performance of recommended models. Include:
            1. Appropriate visualization types for the problem (e.g., ROC curves, confusion matrices)
            2. Code snippets for generating visualizations
            3. Guidelines for interpreting the visualizations""",
            agent=Visualization_Agent,
            expected_output="Code snippets and explanations for model comparison visualizations.",
        )

        crew = Crew(
            agents=[
                Problem_Definition_Agent,
                Data_Assessment_Agent,
                AI_Technique_Recommendation_Agent,
                Code_Generator_Agent,
                Cross_Validation_Agent,
                Visualization_Agent,
            ],
            tasks=[
                task_define_problem,
                task_assess_data,
                task_recommend_technique,
                task_generate_code,
                task_cross_validation,
                task_visualize_comparison,
            ],
            verbose=False,
        )

        results = crew.kickoff()

        # Write the output to output.md
        with open('output.md', "w") as file:
            file.write("# AI Project Summary\n\n")
            file.write("## Problem Statement\n")
            file.write(f"{user_question}\n\n")

            for task in crew.tasks:
                if task.agent.role != "Problem_Definition_Agent":
                    file.write(f"## {task.agent.role.replace('_', ' ')}\n\n")
                    file.write(f"**Task Summary:** {task.output.summary}\n\n")
                    file.write("**Key Recommendations:**\n")
                    key_points = extract_key_points(task.output.raw)
                    if key_points:
                        for point in key_points:
                            file.write(f"{point}\n")
                    else:
                        file.write("No specific key points extracted. Please refer to the task summary.\n")
                    file.write("\n")
                    
                    # Extract and write code snippets
                    code_snippets = extract_code_snippets(task.output.raw)
                    if code_snippets:
                        file.write("**Code Snippets:**\n")
                        for snippet in code_snippets:
                            file.write(f"```python\n{snippet}\n```\n\n")

            file.write("## Next Steps\n")
            file.write("Based on the analysis, consider the following next steps:\n")
            next_steps = extract_key_points(crew.tasks[-1].output.raw)
            if next_steps:
                for step in next_steps:
                    file.write(f"- {step}\n")
            else:
                file.write("- Implement the suggested preprocessing steps\n")
                file.write("- Train and evaluate the recommended AI techniques\n")
                file.write("- Fine-tune the best performing approach\n")
                file.write("- Deploy the model/system and monitor its performance\n")

        print("Complete output has been exported to output.md")
        
        # Print the content of the file after writing
        with open('output.md', 'r') as file:
            print(f"Content of output.md:\n{file.read()}")

def extract_key_points(raw_output):
    key_points = []
    lines = raw_output.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('*') or (': ' in line and not line.startswith('**')):
            key_points.append(line)
    return key_points

def extract_code_snippets(raw_output):
    code_snippets = []
    lines = raw_output.split('\n')
    in_code_block = False
    current_snippet = []
    
    for line in lines:
        if line.strip().startswith('```python'):
            in_code_block = True
            current_snippet = []
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            code_snippets.append('\n'.join(current_snippet))
        elif in_code_block:
            current_snippet.append(line)
    
    return code_snippets

if __name__ == "__main__":
    main()
