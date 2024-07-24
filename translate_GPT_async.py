import os
import argparse
import nbformat
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("The environment variable 'OPENAI_API_KEY' is not set.")

client = AsyncOpenAI(api_key=api_key)

async def translate_text(text):
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Translate the following English text to Chinese: {text}"}
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        logger.debug(traceback.format_exc())
        return text  # Return original text if translation fails

async def translate_notebook(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    try:
        # Load the notebook
        with open(input_file, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)

        # Create translation tasks for each Markdown cell
        translation_tasks = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                task = asyncio.create_task(translate_text(cell['source']))
                translation_tasks.append(task)
        
        # Wait for all translation tasks to complete
        translated_texts = await asyncio.gather(*translation_tasks)
        
        # Assign translated texts to cells
        for cell, translated_text in zip([c for c in notebook['cells'] if c['cell_type'] == 'markdown'], translated_texts):
            cell['source'] = translated_text

        # Write the translated notebook to a new file
        with open(output_file, 'w', encoding='utf-8') as file:
            nbformat.write(notebook, file)
        
        logger.info(f"Successfully translated notebook: {input_file}")
    except Exception as e:
        logger.error(f"Error processing notebook {input_file}: {str(e)}")
        logger.debug(traceback.format_exc())

def find_notebooks(directory):
    notebooks = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".ipynb"):
                    notebooks.append(os.path.join(root, file))
        return notebooks
    except Exception as e:
        logger.error(f"Error finding notebooks: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

async def main(directory):
    notebooks = find_notebooks(directory)
    if not notebooks:
        logger.warning(f"No notebooks found in directory: {directory}")
        return

    tasks = []
    for notebook in notebooks:
        output_file = notebook.replace('.ipynb', '_CN.ipynb')
        tasks.append(asyncio.create_task(translate_notebook(notebook, output_file)))

    await asyncio.gather(*tasks)
    logger.info("Translation process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Jupyter Notebooks in a directory.")
    parser.add_argument("directory", type=str, help="Directory to search for .ipynb files")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.directory))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        logger.debug(traceback.format_exc())