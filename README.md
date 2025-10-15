# ICRT-jailbreaking
## Third-Party Components and Licensing
This project incorporates prompt template files within the `templates` directory, which are derived from the following external repository. We acknowledge and thank the original author for their work.

- **Source Project:** [ICRT](https://github.com/longlong-www/ICRT.git)
- **Original Creator:** [longlong-www]
- **Original License:** This portion of the code is covered under the **GNU Affero General Public License v3.0**.

## Citation
The jailbreaking method implemented in this repository is based on the methodology in the following academic publication:


## Project Structure
This project is designed to achieve ICRT-jailbreaking method, and its structure is organized as follows:
* **'templates/'**: Contains the prompt templates used for processing the original harmful questions.
* **'code/'**: Houses the pipeline script and core execution logic for the project.

## Getting Started
**Prerequisties**
To run this project successfully, you will need the following environment and credentials:
1. **Python Environment**: Python 3.8 or a later version is recommended.
2. **OpenAI Library**: The script relies on the official `openai` Python library for API calls.
3. **API Key & Base URL**: You need access to an OpenAI-compatible API service. This requires:
    - An **API Key**
    - A **Base URL** pointing to your specific API service endpoint.

**Installation**
Please install the required Python library using pip:
```text
pip install openai
```

**Configuration**
Before your execution, you need to configure your API credentials.
1. open the `test.py` file.
2. Locate the `OpenAI` client initialization near the top of the file.
```Python
client = OpenAI(
    api_key="",  # <-- Enter your API Key here
    base_url=""  # <-- Enter your Base URL here
)
```
3. Replace the placeholder values for `api_key` and `base_url` with your actual credentials.(**Note**: `gpt-3.5-turbo-ca` for the auxiliary model and `gpt-4o-ca` for the target model are custom names. Ensure that the service pointed to by your base_url recognizes and provides access to these models.)

**Running the Script**
The script is designed to process an input file in JSON Lines format and write the results to a separate output file.
1. **Prepare your dataset input file**
2. **Set File Paths**
3. **Excute**(make sure you have installed openai)
```text
git clone git@github.com:Yandick/ICRT-jailbreaking.git && cd ICRT-jailbreaking/code
python test.py
```