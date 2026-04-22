from kfp import dsl
from kfp import compiler

# =============================================================================
# 🧮 COMPONENT DEFINITIONS
# =============================================================================

@dsl.component(base_image='python:3.9-slim')
def add_numbers(a: int, b: int) -> int:
    """Calculate the sum of two integers.
    
    Args:
        a: First integer
        b: Second integer
    Returns:
        Sum of a and b
    """
    print(f"Calculating {a} + {b}")
    return a + b


@dsl.component(base_image='python:3.9-slim')
def multiply_numbers(a: int, b: int) -> int:
    """Calculate the product of two integers.
    
    Args:
        a: First integer
        b: Second integer
    Returns:
        Product of a and b
    """
    print(f"Calculating {a} * {b}")
    return a * b

# =============================================================================
# 📐 PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name='arithmetic-demonstration-pipeline',
    description='A fundamental pipeline illustrating sequential task execution and data routing.'
)
def simple_math_pipeline(x: int = 5, y: int = 10, multiplier: int = 2):
    """Chain addition and multiplication tasks.
    
    Flow: x + y → (result) * multiplier
    """
    # Run addition first
    add_task = add_numbers(a=x, b=y)
    
    # Pass addition result to multiplication
    multiply_task = multiply_numbers(a=add_task.output, b=multiplier)

# =============================================================================
# ⚙️ COMPILATION
# =============================================================================

if __name__ == '__main__':
    print("Compiling the pipeline into a Kubeflow-compatible YAML schema...")
    compiler.Compiler().compile(
        pipeline_func=simple_math_pipeline,
        package_path='compiled_math_pipeline.yaml'
    )
    print("Compilation complete! ✅")
