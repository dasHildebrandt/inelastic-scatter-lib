# Setup
## 1. Cloning Repository
Run the git clone command in your project directory. 

``git clone https://github.com/dasHildebrandt/inelastic-scatter-lib.git``


## 2. Setup Virtual Environment

The Poetry package is used to manage project dependencies.
If you haven't already, you need to install Poetry on your system. You can use pip to install it:

``pip install poetry``

#### Activate the Virtual Environment
Poetry automatically creates a virtual environment for your project when you initialize it. To activate the virtual environment, use the following command:

``poetry shell``
This will activate the virtual environment and change your shell prompt to indicate that you are in the virtual environment.

#### Install Dependencies
Use Poetry to add dependencies to your project. For example, to add the requests library, run:

``poetry add requests``

Poetry will manage the dependencies in the pyproject.toml file.

#### Run Your Python Scripts

You can now run your Python scripts as you normally would while inside the Poetry-managed virtual environment. Any dependencies you added with Poetry will be available.

Exit the Virtual Environment:
When you're done working on your project, you can exit the virtual environment by simply running:

``exit``
This will return you to your system's global Python environment.

#### Lock Dependencies (Optional)
Poetry also allows you to lock your project's dependencies and generate a pyproject.lock file by running:

``poetry lock``
This is especially useful when you want to ensure that other developers working on your project use the same dependencies.

Install Dependencies from pyproject.toml (Optional):
If you have a pyproject.toml file and want to install the project's dependencies on another machine or for someone else, you can run:

``poetry install``
This command will install the dependencies specified in the pyproject.toml file.

