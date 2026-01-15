"""
SWMH 数据集分类提示词构建。
"""

import json
from typing import List


def build_classification_prompt(conversation_input: List[str], dataset_id: str) -> str:
    """
    构建 SWMH 数据集的分类提示词。
    
    Args:
        conversation_input: 输入文本列表（通常只有一个元素，即帖子文本）
        dataset_id: 数据集标识符
        
    Returns:
        格式化的提示词字符串
    """
    # 将输入文本合并
    text = "\n".join(conversation_input) if conversation_input else ""
    
    prompt = f"""You are an expert mental health AI specialized in analyzing Reddit posts from mental health-related subreddits. Your task is to analyze a user's post and identify whether it reflects a possible mental health crisis or disorder. If so, classify it into one of the following categories.

This dataset (SWMH - Reddit SuicideWatch and Mental Health Collection) contains posts from mental health-related subreddits, including discussions about suicide-related intention and mental disorders like depression, anxiety, and bipolar disorder.

Below are definitions for classification. Choose ONE category that best matches the overall post. Return your answer as a JSON object like this:
{{
  "dataset_id": "{dataset_id}",
  "label": "<one of the categories below>",
  "explanation": "<brief explanation>"
}}

### CATEGORIES

The dataset contains posts from 5 subreddits. Choose the category that best matches the post:

- **depression**: Posts from the depression subreddit. The user shows signs of depression, including persistent sadness, loss of interest, feelings of worthlessness, fatigue, difficulty concentrating, or changes in sleep/appetite. The post reflects a depressed mood that affects daily functioning.
  Examples:
    "I've been feeling so down for weeks, nothing brings me joy."
    "I can't get out of bed, everything feels pointless."
    "I'm so tired all the time, even simple tasks feel impossible."

- **suicidal**: Posts from the SuicideWatch subreddit. The user expresses thoughts about wanting to die or end their life, either passively ("I wish I didn't exist") or actively ("I've thought about how to kill myself"). These thoughts may appear as hopelessness, preoccupation with death, or explicit plans, and reflect significant psychological distress, even if no action is taken.
  Examples:
    "I wish I didn't exist anymore."
    "I've been thinking about ways to end it all."
    "Sometimes I wonder if everyone would be better off without me."
    "I don't see the point in living anymore."
    "I feel deader than dead."

- **anxiety**: Posts from the Anxiety subreddit. The user is experiencing anxiety symptoms, including excessive worry, restlessness, difficulty concentrating, irritability, muscle tension, or panic attacks. The post reflects ongoing anxiety that impacts their life.
  Examples:
    "I can't stop worrying about everything."
    "I had a panic attack in the middle of class."
    "My anxiety is so bad I can't leave the house."

- **bipolar**: Posts from the bipolar subreddit. The user shows signs of bipolar disorder, including mood swings between depression and mania/hypomania, periods of high energy, impulsivity, or rapid cycling between emotional states.
  Examples:
    "I go from feeling amazing to completely hopeless."
    "I had so much energy last week, but now I can't move."
    "My moods swing so wildly I don't know what's normal."

- **offmychest**: Posts from the offmychest subreddit. These posts typically do not show clear signs of mental health crisis or disorder. They may discuss general life issues, venting, or personal experiences without indicating a mental health crisis.
  Examples:
    General venting about life situations
    Personal experiences without mental health crisis indicators
    Discussions that don't clearly fit into the above categories

### INSTRUCTIONS

1. Only choose ONE label that best fits the overall post.
2. Only classify the post shown below.
3. Pay attention to the context and severity of the mental health concerns expressed.

IMPORTANT: You MUST only classify the post shown below. Do not reuse or return any other previous content. ONLY return the exact JSON response for this post.

Post:
{text}
"""
    return prompt

