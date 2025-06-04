# classification.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Callable
import argparse
from transformers import pipeline
import gradio as gr


def get_topic_classification_pipeline() -> Callable[[str], dict]:
    """
    Question:
        Load the pipeline for topic text classification.
        There are 10 possible labels: 
            'Society & Culture', 'Science & Mathematics', 'Health',
            'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
            'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
        Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
        Notice that we have time limits so you should not use a model that is too large. A model with 
        100M params is enough.

    Returns:
        func (Callable): A function that takes a string as input and returns a dictionary with the
        predicted label and its score.

    Example:
        >>> func = get_topic_classification_pipeline()
        >>> result = func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")
        {"label": "Politics & Government", "score": 0.9999999403953552}
    """
    # 题目要求的10个标签
    target_labels = [
        'Society & Culture', 'Science & Mathematics', 'Health',
        'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
        'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
    ]
    
    # 映射从Yahoo Answers Topics的数字类别到目标标签
    # 根据数据集信息映射
    # 0: Society & Culture
    # 1: Science & Mathematics
    # 2: Health
    # 3: Education & Reference
    # 4: Computers & Internet
    # 5: Sports
    # 6: Business & Finance
    # 7: Entertainment & Music
    # 8: Family & Relationships
    # 9: Politics & Government
    label_map = {
        0: 'Society & Culture',
        1: 'Science & Mathematics', 
        2: 'Health',
        3: 'Education & Reference', 
        4: 'Computers & Internet', 
        5: 'Sports', 
        6: 'Business & Finance',
        7: 'Entertainment & Music', 
        8: 'Family & Relationships', 
        9: 'Politics & Government'
    }
    
    # 使用预训练模型，选择一个针对Yahoo Answers Topics数据集微调过的模型
    # 不设置top_k，这样会返回单个最高分数的结果
    pipe = pipeline("text-classification", model="fabriceyhc/bert-base-uncased-yahoo_answers_topics")

    def func(text: str) -> dict:
        # 使用模型进行预测
        results = pipe(text)

        # 处理返回格式 - pipeline默认返回列表
        if isinstance(results, list) and len(results) > 0:
            result = results[0]  # 取第一个（最高分数的）结果
        else:
            result = results

        # 确保result是字典类型
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict result, got {type(result)}: {result}")

        # 获取预测的类别ID和分数
        pred_id = int(result["label"].split('_')[-1])  # 从LABEL_X中提取X
        score = result["score"]

        # 映射到目标标签
        label = label_map[pred_id]

        return {"label": label, "score": score}
    
    return func


def main():
    parser = argparse.ArgumentParser(description="Topic Classification Pipeline")
    parser.add_argument("--task", type=str, help="Task name", choices=["sentiment", "topic"], default="sentiment")
    parser.add_argument("--use-gradio", action="store_true", help="Use Gradio for UI")

    args = parser.parse_args()

    if args.use_gradio and args.task == "sentiment":
        # Example usage with Gradio
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        iface = gr.Interface.from_pipeline(pipe)
        iface.launch()

    elif args.use_gradio and args.task == "topic":
        # Visualize the topic classification pipeline with Gradio
        pipe = get_topic_classification_pipeline()
        iface = gr.Interface(
            fn=lambda x: {item["label"]: item["score"] for item in [pipe(x)]},
            inputs=gr.components.Textbox(label="Input", render=False),
            outputs=gr.components.Label(label="Classification", render=False),
            title="Text Classification",
        )
        iface.launch()

    elif not args.use_gradio and args.task == "sentiment":
        # Example usage
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        print(pipe("This movie is great!")[0]) # {'label': 'positive', 'score': 0.988831102848053}

    elif not args.use_gradio and args.task == "topic":
        # Test the function
        func = get_topic_classification_pipeline()
        print(func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")) # {"label": "Politics & Government", "score": ...}


if __name__ == "__main__":
    main()
