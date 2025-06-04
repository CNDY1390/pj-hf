# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).


import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM
from enum import Enum
from typing import List, Dict, Tuple, Any, Optional, Union


class InvalidTreeError(Exception):
    pass


class TokenType(Enum):
    """枚举类型定义不同的标记类型"""
    BOS = "BOS"       # 序列开始标记 <s>
    EOS = "EOS"       # 序列结束标记 </s>
    ONT = "ONT"       # 开括号非终结符 (X
    CNT1 = "CNT1"     # 闭括号非终结符类型1 X)
    CNT2 = "CNT2"     # 闭括号非终结符类型2 X) (复制的)
    TERM = "TERM"     # 终端符号


class TreeValidator:
    """语法树验证器，负责检查动作序列是否构成有效的语法树"""
    
    @staticmethod
    def validate(actions: List[str]) -> None:
        """
        验证动作序列是否构成有效的语法树
        
        Args:
            actions: 动作序列
            
        Raises:
            InvalidTreeError: 如果序列无效
        """
        # 基本检查
        TreeValidator._check_basic_requirements(actions)
        
        # 括号平衡检查
        TreeValidator._check_bracket_balance(actions)
        
        # 树结构检查
        TreeValidator._check_tree_structure(actions)
      @staticmethod
    def _check_basic_requirements(actions: List[str]) -> None:
        """检查序列的基本要求"""
        if not actions or len(actions) < 2: raise InvalidTreeError("序列太短")
        if actions[0] != "<s>" or actions[-1] != "</s>": raise InvalidTreeError("序列必须以<s>开始并以</s>结束")
      @staticmethod
    def _check_bracket_balance(actions: List[str]) -> None:
        """检查括号是否平衡匹配"""
        balance = 0
        
        for idx, token in enumerate(actions):
            if (idx == 0 and token == "<s>") or (idx == len(actions) - 1 and token == "</s>"): continue
                
            balance += 1 if token.startswith("(") else (-1 if token.endswith(")") else 0)
                
            if balance < 0: raise InvalidTreeError(f"在开括号前关闭非终结符: {token}，位置 {idx}")
                
        if balance != 0: raise InvalidTreeError(f"序列末尾存在不匹配的非终结符。平衡值: {balance}")
      @staticmethod
    def _check_tree_structure(actions: List[str]) -> None:
        """检查树结构的有效性，包括内容检查"""
        stack = []; has_terminal = False
        
        for idx, token in enumerate(actions):
            if token in ["<s>", "</s>"]: continue
                
            if token.startswith("("):
                # [非终结符名称, 是否有内容]
                stack.append([token[1:], False])
            elif token.endswith(")"):
                if not stack: raise InvalidTreeError(f"关闭非终结符 '{token}' 没有匹配的开括号")
                    
                nt_name, has_content = stack.pop()
                
                if nt_name != token[:-1]: raise InvalidTreeError(f"非终结符不匹配: 预期 '{nt_name})', 得到 '{token}'")
                if not has_content: raise InvalidTreeError(f"非终结符 '{nt_name}' 为空或只包含空非终结符")
                    
                # 标记父级有内容
                if stack: stack[-1][1] = True
            else:  # 终端符号
                has_terminal = True
                if not stack: raise InvalidTreeError(f"在任何非终结符之外找到终端 '{token}'")
                    
                # 标记当前非终结符及其所有祖先有内容
                stack[-1][1] = True; [stack[i].__setitem__(1, True) for i in range(len(stack) - 1)]
        
        # 检查是否有终端符号
        if not has_terminal and len(actions) > 2:
            is_just_bos_eos = len(actions) == 2 and actions[0] == "<s>" and actions[1] == "</s>"
            if not is_just_bos_eos: raise InvalidTreeError("动作序列中没有找到终端符号")


class SequenceProcessor:
    """序列处理器，负责处理输入、输出和位置ID"""
      @staticmethod
    def process_sequence(actions: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """
        处理动作序列，生成输入、输出和位置ID
        
        Args:
            actions: 动作序列
            
        Returns:
            inputs: 处理后的输入序列
            labels: 处理后的输出序列
            position_ids: 位置ID序列
        """
        inputs = []; labels = []; position_ids = []; depth = 0
        
        for token in actions:
            if token == "<s>":
                inputs.append(token); labels.append(token); position_ids.append(0); depth = 0
            elif token.startswith("("):
                inputs.append(token); labels.append(token); position_ids.append(depth); depth += 1
            elif token.endswith(")"):
                depth -= 1
                # 原始闭括号
                inputs.append(token); labels.append(token); position_ids.append(depth)
                # 复制的闭括号
                inputs.append(token); labels.append("<pad>"); position_ids.append(depth)
            elif token == "</s>":
                inputs.append(token); labels.append(token); position_ids.append(0)
            else:  # 终端符号
                inputs.append(token); labels.append(token); position_ids.append(depth)
                
        return inputs, labels, position_ids


class AttentionMaskGenerator:
    """注意力掩码生成器，负责生成STACK/COMPOSE注意力掩码"""
      @staticmethod
    def get_token_types(inputs: List[str], labels: List[str]) -> List[TokenType]:
        """
        确定每个输入标记的类型
        
        Args:
            inputs: 输入序列
            labels: 标签序列
            
        Returns:
            token_types: 标记类型列表
        """
        token_types = []
        
        for token, label in zip(inputs, labels):
            token_types.append(
                TokenType.BOS if token == "<s>" else 
                TokenType.EOS if token == "</s>" else 
                TokenType.ONT if token.startswith("(") else 
                (TokenType.CNT1 if label != "<pad>" else TokenType.CNT2) if token.endswith(")") else 
                TokenType.TERM
            )
                
        return token_types
      @staticmethod
    def generate_attention_mask(inputs: List[str], labels: List[str]) -> torch.Tensor:
        """
        生成注意力掩码
        
        Args:
            inputs: 输入序列
            labels: 标签序列
            
        Returns:
            attention_mask: 注意力掩码张量
        """
        token_types = AttentionMaskGenerator.get_token_types(inputs, labels)
        seq_len = len(inputs)
        attention_mask = torch.zeros(seq_len, seq_len, dtype=torch.float)
        stack = []
        
        for i in range(seq_len):
            current_type = token_types[i]
            
            if current_type == TokenType.EOS: continue
                
            if current_type == TokenType.CNT1:
                # COMPOSE注意力模式
                j = i
                while j < seq_len and token_types[j] != TokenType.ONT:
                    attention_mask[i, j] = 1.0; j = stack.pop()
                attention_mask[i, j] = 1.0; stack.append(i)
            else:
                # STACK注意力模式
                if current_type != TokenType.CNT2: stack.append(i)
                [attention_mask.__setitem__((i, attended_idx), 1.0) for attended_idx in stack]
                attention_mask[i, i] = 1.0
                
        return attention_mask

def mapping_function(example: dict) -> dict:
    """
    Question:
        Your task is to return the processed input, processed output, attention mask, and absolute positions of the action sequence for valid actions sequence. The following order may be your implementation order:

            1. Check whether the given action sequence is a valid sequence to generate a legal parse tree. If it is invalid, please raise an InvalidTreeError Exception.
            2. The processed input: a list of strings. It should duplicate all closing nonterminals in the given action sequence.
            3. The processed output: a list of strings. It should insert '<pad>' after all closing nonterminals in the given action sequence.
            4. The absolute positions: a list of integers. The absolute position of each token is defined as the depth of it in the tree.
            5. The attention mask: a 2d torch tensor. This is the attention mask with STACK/COMPOSE attention. The attention mask of '</s>' is all 0s.

        HINT: It is guaranteed that the first item of input is '<s>' (beginning of sequence), and the last item of input is '</s>' (end of sequence). The absolute positions of both '<s>' and '</s>' are 0 in this question.
    
    Args:
        example (dict): The example to process. It has the following fields:
            - actions (List[str]): The action sequence. It is a list of strings which can be regarded as an action sequence for generative transition-based parsing.

    Return:
        mapped (dict): The mapped example. It has the following fields:
            - inputs (List[str]): The processed input. A list of tokens for the input.
            - labels (List[str]): The processed output. A list of tokens for the expected output.
            - position_ids (List[int]): The absolute positions. A list of integers representing the absolute position of each token in the input.
            - attention_mask (torch.Tensor): The attention mask. Shape: (len(input), len(input)). A 2D tensor representing the attention mask for the input sequence. 1 for valid tokens, 0 for padding tokens.

    Example:
        >>> mapping_function({"actions": ["<s>", "(S", "(NP", "the", "blue", "bird", "NP)", "(VP", "sings", "VP)", "S)", "</s>"]})
        {
            'inputs': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>'],
            'labels': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '<pad>', '(VP', 'sings', 'VP)', '<pad>', 'S)', '<pad>', '</s>'],
            'position_ids': [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 0, 0],
            'attention_mask': tensor([[...]])
        }
    """

    """YOUR CODE HERE"""
    actions = example["actions"]
    
    # 1. 验证动作序列
    TreeValidator.validate(actions)
    
    # 2-4. 处理序列，生成输入、输出和位置ID
    inputs, labels, position_ids = SequenceProcessor.process_sequence(actions)
    
    # 5. 生成注意力掩码
    attention_mask = AttentionMaskGenerator.generate_attention_mask(inputs, labels)
    
    return {
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    """
    Question:
        Create a Trainer object for the model. The Trainer is used to train the model on the dataset.
        Select the appropriate training arguments for the Trainer. For example, setting the proper learning rate,
        batch size, optimizer, learning rate scheduler, number of epochs, etc. would be a good idea.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for the model.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The dataset to train on.

    Returns:
        trainer (Trainer): The Trainer object for the model.

    Example:
        >>> trainer = get_trainer(tokenizer, model, train_dataset)
        >>> trainer.train()
        >>> trainer.evaluate(train_dataset)
        {'eval_loss': 2.1234, ...}
    """

    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch
    
    """YOUR CODE HERE"""
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    
    # 创建Trainer
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent] for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": [[mask] for mask in examples["attention_mask"]],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
