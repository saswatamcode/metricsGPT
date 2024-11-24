# parse_requirements.py
import toml

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

# Read the pyproject.toml file
with open("pyproject.toml", "r", encoding="utf-8") as pyproject_file:
    pyproject = toml.load(pyproject_file)

# Update dependencies in pyproject.toml
pyproject["project"]["dependencies"] = requirements

# Write updated pyproject.toml back to file
with open("pyproject.toml", "w", encoding="utf-8") as pyproject_file:
    toml.dump(pyproject, pyproject_file)

print("Updated pyproject.toml with requirements.txt content!")