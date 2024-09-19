import os
import nbformat
import asyncio
from openai import AsyncOpenAI
import logging
import traceback
import typer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("环境变量 'OPENAI_API_KEY' 未设置。")

client = AsyncOpenAI(api_key=api_key)

async def translate_text(text):
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"将以下英文文本翻译成中文：{text}"}
            ],
            #model="gpt-3.5-turbo",
            model="gpt-4o-mini",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"翻译文本时出错：{str(e)}")
        logger.debug(traceback.format_exc())
        return text  # 如果翻译失败，返回原文

async def translate_notebook(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在：{input_file}")
        return

    try:
        # 加载笔记本
        with open(input_file, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)

        # 为每个 Markdown 单元格创建翻译任务
        translation_tasks = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                task = asyncio.create_task(translate_text(cell['source']))
                translation_tasks.append(task)
        
        # 等待所有翻译任务完成
        translated_texts = await asyncio.gather(*translation_tasks)
        
        # 将翻译后的文本分配给单元格
        for cell, translated_text in zip([c for c in notebook['cells'] if c['cell_type'] == 'markdown'], translated_texts):
            cell['source'] = translated_text

        # 将翻译后的笔记本写入新文件
        with open(output_file, 'w', encoding='utf-8') as file:
            nbformat.write(notebook, file)
        
        logger.info(f"成功翻译笔记本：{input_file}")
    except Exception as e:
        logger.error(f"处理笔记本 {input_file} 时出错：{str(e)}")
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
        logger.error(f"查找笔记本时出错：{str(e)}")
        logger.debug(traceback.format_exc())
        return []

async def process_notebooks(directory: str):
    notebooks = find_notebooks(directory)
    if not notebooks:
        logger.warning(f"在目录 {directory} 中未找到笔记本")
        return

    tasks = []
    for notebook in notebooks:
        output_file = notebook.replace('.ipynb', '_CN.ipynb')
        tasks.append(asyncio.create_task(translate_notebook(notebook, output_file)))

    await asyncio.gather(*tasks)
    logger.info("翻译过程完成")

app = typer.Typer()

@app.command()
def main(directory: str = typer.Argument(..., help="要搜索 .ipynb 文件的目录")):
    """
    翻译指定目录中的 Jupyter 笔记本。

    此脚本通过异步协程操作提高效率，使用 OpenAI 的 GPT-3.5 模型将 Jupyter 笔记本中的 Markdown 单元格从英文翻译成中文。使用步骤：

    1. 安装 `openai`、`nbformat` 和 `tqdm` 库。

    2. 在 `client` 中设置 OpenAI API 密钥。

    3. 输入待翻译的 `.ipynb` 文件路径。

    4. 脚本将创建一个翻译后的副本，文件名后加 "_CN.ipynb"。

    5. 使用 `python3 translate_GPT_async.py` 运行脚本进行并行翻译。

    """
    try:
        asyncio.run(process_notebooks(directory))
    except KeyboardInterrupt:
        logger.info("用户中断进程")
    except Exception as e:
        logger.critical(f"主执行过程中出现严重错误：{str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    app()