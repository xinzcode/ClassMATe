import pdfplumber
import os
import re
from collections import defaultdict

def clean_text(text):
    # 清理多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 处理换行符
    text = text.replace('\n', ' ')
    # 处理多个空格
    text = re.sub(r' +', ' ', text)
    return text

def pdf_to_txt(pdf_folder, txt_folder):
    # 检查保存 txt 文件的文件夹是否存在，如果不存在则创建
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # 创建一个字典来存储以相同数字开头的文件内容
    grouped_texts = defaultdict(str)

    # 遍历 pdf_folder 中的所有 PDF 文件
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # 提取文件名的第一个数字
            match = re.match(r'^(\d)', filename)
            if match:
                group_key = match.group(1)  # 获取第一个数字
            else:
                group_key = "no_number"  # 没有以数字开头的文件放在这个组

            full_text = ""
            try:
                # 打开 PDF 并提取内容
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            # 检查页面是否包含图像并跳过图像
                            if page.images:
                                print(f"Page {page_num + 1} of {filename} contains images, ignoring images.")

                            # 提取文本并保留页面的布局
                            page_text = page.extract_text(layout=True)
                            if page_text:
                                full_text += page_text + "\n"  # 添加换行符保持页面结构
                        except Exception as e:
                            print(f"Error extracting text from page {page_num + 1} of {filename}: {e}")

                # 将提取的文本添加到分组文本中，用三个换行符分隔
                grouped_texts[group_key] += full_text + "\n\n\n"
                print(f"Processed: {filename} (Group: {group_key})")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # 将每个组的文本写入单独的 txt 文件
    for group_key, combined_text in grouped_texts.items():
        cleaned_text = combined_text
        txt_path = os.path.join(txt_folder, f"{group_key}.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(cleaned_text.strip())  # 写入时去除最后多余的换行符
        print(f"Saved group {group_key} to {txt_path}")


# 使用方法
pdf_folder = "pdf"  # 替换为PDF文件夹路径
txt_folder = "txt"  # 替换为TXT保存文件夹路径
pdf_to_txt(pdf_folder, txt_folder)
