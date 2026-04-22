from kfp import dsl
from kfp import compiler
from kfp import kubernetes

@dsl.component(base_image='python:3.9-slim')
def write_test_file(message: str, file_name: str = "test_log.txt"):
    import os
    from datetime import datetime
    
    # The mount path we will define in the pipeline
    mount_path = "/mnt/data"
    full_path = os.path.join(mount_path, file_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(mount_path, exist_ok=True)
    
    # Append the message with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(full_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    
    print(f"Successfully wrote to {full_path}")
    
    # List files to verify write in logs
    print(f"Current files in {mount_path}: {os.listdir(mount_path)}")

@dsl.pipeline(
    name='volume-test-pipeline',
    description='A simple pipeline to verify PVC write permissions.'
)
def volume_test_pipeline(message: str = "Hello from Kubeflow!"):
    
    # 1. Instantiate the task
    write_task = write_test_file(message=message)

    # 2. Mount the PVC using the kubernetes extension
    # IMPORTANT: Ensure 'shared-storage-pvc' exists in your cluster
    kubernetes.mount_pvc(
        task=write_task,
        pvc_name='tutorial',
        mount_path='/mnt/data'
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=volume_test_pipeline,
        package_path='volume_test_pipeline.yaml'
    )