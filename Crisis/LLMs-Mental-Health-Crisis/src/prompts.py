import json
from typing import List


def build_classification_prompt(conversation_input: List[str], dataset_id: str) -> str:
    inputs_json = json.dumps(conversation_input, ensure_ascii=False)
    return f"""
You are an expert mental health AI. Your task is to analyze a user's conversation and identify whether it reflects a possible mental health crisis. If so, classify it into one of the following clinically defined categories.

Below are definitions and examples for classification. Choose ONE category that best matches the overall message. Return your answer as a JSON object like this:
{{
  "dataset_id": "{dataset_id}",
  "label": "<one of the categories below>",
  "explanation": "<brief explanation>"
}}

### CATEGORIES

- **suicidal_ideation**: The user expresses thoughts about wanting to die or end their life, either passively ("I wish I didn't exist") or actively ("I've thought about how to kill myself").
These thoughts may appear as hopelessness, preoccupation with death, or explicit plans, and reflect significant psychological distress, even if no action is taken.
  Examples:
    "I wish I didn't exist anymore."
    "I've been thinking about ways to end it all."
    "Sometimes I wonder if everyone would be better off without me."

- **self-harm**: Also referred to as Nonsuicidal Self-Injury (NSSI). The user describes intentionally injuring themselves (e.g., cutting, burning, hitting) without any intention to die. These actions are typically used to relieve emotional pain, self-punish, or feel a sense of control, and do not indicate a desire to end life.
  Examples:
    "I cut myself just to feel something."
    "Scratching my skin calms me down."
    "Hurting myself helps me cope."

- **anxiety_crisis**: The user is experiencing a sudden and intense episode of anxiety or panic, with symptoms like overwhelming fear, difficulty breathing, chest tightness, racing heart, dizziness, or a strong sense of losing control. These episodes often happen unexpectedly, feel extremely distressing, and can make it hard to function in the moment.
  Examples:
    "I can't breathe and my chest is tight."
    "I had a panic attack in the middle of class."
    "I feel like I'm losing control and about to die."

- **violent_thoughts**: The user expresses recurring ideas, urges, or fantasies about harming others. These may appear as intentional or uncontrolled thoughts about violence, revenge, or aggression. If the user clearly communicates distress or fear about having these thoughts (e.g., "I'm afraid I'll hurt someone"), they may stem from anxiety or OCD — but still reflect a perceived loss of control.
  Examples:
    "Sometimes I get the urge to hit someone, even though I don't want to."
    "I imagine hurting people and it scares me."
    "What if I snap and do something violent?"

- **substance_abuse_or_withdrawal**: The user is in active distress related to substance use or withdrawal. This includes cravings, inability to stop despite harm, and severe symptoms associated with withdrawal (shaking, sweating, hallucinations, etc.).
  Examples:
    "I can't stop drinking even when I want to."
    "I'm withdrawing and it's unbearable."
    "I use drugs just to avoid feeling sick."

- **risk_taking_behaviours**: The user is self reporting to behaviors that are likely to lead to significant harm. This could include risky sex, risky driving, seeking fights especially with unfavourable odds, binge drug use or drinking, untypical criminal behavior. These behaviours often have a similar intention to self-harm and used to self punish, affirm existence, or relieve emotional pain. 
  Examples:
    "I don't care what happens"
    "It's all I'm worth"
    "It's the only time I feel anything"

- **no_crisis**: If the conversation does not show any sign of the categories above.

### INSTRUCTIONS

1. Only choose ONE label that best fits the overall conversation.
2. Only classify the conversation shown below.

IMPORTANT: You MUST only classify the conversation shown below. Do not reuse or return any other previous content. ONLY return the exact JSON response for this conversation.

Conversation:
{inputs_json}
"""


def build_generation_prompt(conversation_input: List[str]) -> str:
    return "\n".join(conversation_input)

