"""
C-SSRS 数据集分类提示词构建。
"""

from typing import List


def build_classification_prompt(
    conversation_input: List[str],
    dataset_id: str,
    experiment: str = "exp1"
) -> str:
    """
    构建 C-SSRS 数据集的分类提示词。
    使用5-label分类：Supportive、Indicator、Ideation、Behavior、Attempt
    
    Args:
        conversation_input: 输入文本列表（通常只有一个元素，即帖子文本）
        dataset_id: 数据集标识符
        experiment: 保留参数以兼容接口
        
    Returns:
        格式化的提示词字符串
    """
    # 将输入文本合并
    text = "\n".join(conversation_input) if conversation_input else ""
    
    # 使用5-label分类
    labels_desc = """
### CATEGORIES (5-label classification)

Choose ONE category that best matches the user's posts:

1. **Supportive**: This category is defined as individuals engaging in discussion but with no language that expressed any history of being at-risk in the past or the present. Some identified themselves as having background in mental health care, while others did not define their motive for interacting at all (as opposed to a family history). Since posting on Reddit is not itself a risk factor, so we give these users a category with even lower risk than those expressing support with a history of risk factors. Any use of language such as a history of depression, or "I've been there" would re-categorize a user as exhibiting suicidal indicator, ideation, or being at greater risk, depending on the language used.

2. **Indicator**: This category separated those using at-risk language from those actively experiencing general or acute symptoms. Oftentimes, users would engage in conversation in a supportive manner and share personal history while using at-risk words from the clinical lexicon. These users might express a history of divorce, chronic illness, death in the family, or suicide of a loved one, which are risk indicators on the C-SSRS, but would do so relating in empathy to users who expressed ideation or behavior, rather than expressing a personal desire for self-harm. In this case, it was deemed appropriate to flag such users as IN because while they expressed known risk factors that could be monitored they would also count as false positives if they were accepted as individuals experiencing active ideation or behavior.

3. **Ideation**: C-SSRS begins with Suicidal Ideation (ID), which is defined as thoughts of suicide including preoccupations with risk factors such as loss of job, loss of a strong relationship, chronic disease, mental illness, or substance abuse.

4. **Behavior**: This category can be seen to escalate from Suicidal Ideation. A user with suicidal behavior confesses active or historical self-harm, or active planning to commit suicide, or a history of being institutionalized for mental health. Actions include cutting or using blunt force violence (self-punching and head strikes), heavy substance abuse, planning for suicide attempt, or actions involving a means of death (holding guns or knives, standing on ledges, musing over pills or poison, or driving recklessly).

5. **Attempt**: The last category, an Actual Attempt (AT), is defined as any deliberate action that may result in intentional death, be it a completed attempt or not, including but not limited to attempts where a user called for help, changed their mind or wrote a public "good bye" note.
"""
    
    prompt = f"""You are an expert mental health AI specialized in analyzing Reddit posts for suicide risk assessment based on the Columbia-Suicide Severity Rating Scale (C-SSRS). Your task is to analyze a user's posts and classify them into the appropriate suicide risk severity level.

This dataset (C-SSRS) contains posts from Reddit users in mental health-related subreddits, specifically from SuicideWatch and related communities.

{labels_desc}

### INSTRUCTIONS

1. Carefully read and analyze ALL the posts provided for the user.
2. Consider the overall pattern and severity across all posts.
3. Only choose ONE label that best fits the user's overall risk level.
4. Return your answer as a JSON object like this:
{{
  "dataset_id": "{dataset_id}",
  "label": "<one of the categories above>",
  "explanation": "<brief explanation of your classification>"
}}

IMPORTANT: You MUST only classify the posts shown below. Do not reuse or return any other previous content. ONLY return the exact JSON response for this user.

User's Posts:
{text}
"""
    return prompt

