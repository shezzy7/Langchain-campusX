from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{article_name}" with the following specifications:
    Explanation Style : {exp_style}
    EXplanation length : {exp_length}
    1.Mathematical Details:
        - Include relevent mathematial equations if present in paper.
        - Explain the mathematical concepts using simple,intuitive code snippets where applicable.
    2.Analogies:
        - Use relateable analogies to simplify complex ideas.
    If certain information is not available in the paper,respond with: "Insufficient information available" instead of guessing      
    """,
    input_variables=['article_name',"exp_style","exp_length"],
    validate_template=True
)

template.save("template.json")

# we have written code for our template here and after running this code a new file with name template.json will be created in which our template will be stored and we can access our template thorught this json file anywhere we want.