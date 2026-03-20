from kfp import dsl
from kfp import compiler

# 1. Define the first ephemeral component
# The @dsl.component decorator is required to convert a standard Python function 
# into a Kubeflow component. The component is self-contained and executed in a 
# Docker container defined by 'base_image'.
@dsl.component(base_image='python:3.9-slim')
def add_numbers(a: int, b: int) -> int:
    """An isolated task that calculates the sum of two integers."""
    print(f"Calculating {a} + {b}")
    return a + b

# 2. Define the second ephemeral component
# Similarly, this builds another discrete component that runs independently.
@dsl.component(base_image='python:3.9-slim')
def multiply_numbers(a: int, b: int) -> int:
    """An isolated task that calculates the product of two integers."""
    print(f"Calculating {a} * {b}")
    return a * b

# 3. Construct the Directed Acyclic Graph (DAG)
# The @dsl.pipeline decorator defines the workflow structure, linking components
# together by passing outputs to inputs.
@dsl.pipeline(
    name='arithmetic-demonstration-pipeline',
    description='A fundamental pipeline illustrating sequential task execution and data routing.'
)
def simple_math_pipeline(x: int = 5, y: int = 10, multiplier: int = 2):
    # Instantiate the first task
    add_task = add_numbers(a=x, b=y)
    
    # Instantiate the second task. 
    # Passing 'add_task.output' as an argument implicitly informs Kubeflow 
    # that 'multiply_task' must wait for 'add_task' to complete successfully.
    multiply_task = multiply_numbers(a=add_task.output, b=multiplier)

# 4. Compile the specification into the Intermediate Representation (IR)
# The Compiler translates our Python DSL code into a YAML specification
# that the Kubeflow backend understands and executes on Kubernetes.
if __name__ == '__main__':
    print("Compiling the pipeline into a Kubeflow-compatible YAML schema...")
    compiler.Compiler().compile(
        pipeline_func=simple_math_pipeline,
        package_path='compiled_math_pipeline.yaml'
    )
    print("Compilation complete. The file 'compiled_math_pipeline.yaml' is ready for upload.")
