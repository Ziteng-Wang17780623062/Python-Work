"""数据扰动处理模块。

实现三种类型的扰动：
1. 字符级扰动（正则表达式和Python字符串操作实现）
2. 单词级扰动（基于规则的实现）
3. 句子级扰动（基于规则的实现）

严格按照 扰动方法.md 中的方案实现。
"""

import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from tqdm import tqdm

# ============================
# 扰动强度配置（严格按照扰动方法.md表格）
# ============================

PERTURBATION_INTENSITY = {
    "low": {
        "character": {
            "deletion": 0.01,      # 1% 进一步降低
            "insertion": 0.01,     # 1% 进一步降低
            "replacement": 0.01,   # 1% 进一步降低
            "swap": 0.01,          # 1% 进一步降低
            "case_change": 0.04,   # 4% 进一步降低
            "homoglyph": 0.02,     # 2% 进一步降低
            "repetition": 0.02,    # 字符重复概率 进一步降低
        },
        "word": {
            "word_deletion": 0.05,         # 5% (提高，基于随机性)
            "word_insertion": 0.05,        # 5% (提高，基于随机性)
            "word_replacement": 0.08,      # 8% (提高，基于随机性)
            "word_order_shuffle": True,    # 局部(相邻2-3词)
            "spelling_errors": 0.05,       # 5% (提高，基于随机性)
            "morphological_changes": 0.08, # 8% (提高，基于随机性)
            "abbreviation": 0.15,          # 15% (提高，基于随机性)
            "stopword_removal": 0.20,      # 20% (提高，基于随机性)
        },
        "sentence": {
            "sentence_repetition": 1,            # 重复次数 (进一步提升)
            "sentence_order_shuffle": True,      # 局部/全局打乱
            "sentence_insertion": 0.20,          # 20% (进一步提升，用于计算插入数量)
            "voice_conversion": 0.25,            # 25% (提升，虽然已去除随机性但保留参数)
            "negation_perturbation": 0.15,       # 15% (提升，虽然已去除随机性但保留参数)
            "sentence_truncation": 0.0,          # 前(25%)、中(50%)、后(75%) - 对话长度限制
            "punctuation_perturbation": 0.30,    # 30% (进一步提升，用于计算操作数量)
        },
    },
    "medium": {
        "character": {
            "deletion": 0.025,      # 2.5% 进一步降低
            "insertion": 0.025,     # 2.5% 进一步降低
            "replacement": 0.025,   # 2.5% 进一步降低
            "swap": 0.025,          # 2.5% 进一步降低
            "case_change": 0.08,    # 8% 进一步降低
            "homoglyph": 0.05,      # 5% 进一步降低
            "repetition": 0.03,     # 字符重复概率 进一步降低
        },
        "word": {
            "word_deletion": 0.12,         # 12% (提高，基于随机性)
            "word_insertion": 0.12,        # 12% (提高，基于随机性)
            "word_replacement": 0.18,      # 18% (提高，基于随机性)
            "word_order_shuffle": True,    # 局部(相邻2-3词)
            "spelling_errors": 0.12,       # 12% (提高，基于随机性)
            "morphological_changes": 0.18, # 18% (提高，基于随机性)
            "abbreviation": 0.30,          # 30% (提高，基于随机性)
            "stopword_removal": 0.40,      # 40% (提高，基于随机性)
        },
        "sentence": {
            "sentence_repetition": 2,            # 使用高强度参数
            "sentence_order_shuffle": True,      # 局部/全局打乱
            "sentence_insertion": 0.30,          # 使用高强度参数
            "voice_conversion": 0.80,            # 使用高强度参数
            "negation_perturbation": 0.50,       # 使用高强度参数
            "sentence_truncation": 0.1,          # 保持可用
            "punctuation_perturbation": 0.80,    # 使用高强度参数
        },
    },
    "high": {
        "character": {
            "deletion": 0.04,       # 4% 进一步降低
            "insertion": 0.04,      # 4% 进一步降低
            "replacement": 0.04,    # 4% 进一步降低
            "swap": 0.04,           # 4% 进一步降低
            "case_change": 0.15,    # 15% 进一步降低
            "homoglyph": 0.08,      # 8% 进一步降低
            "repetition": 0.04,     # 字符重复概率 进一步降低
        },
        "word": {
            "word_deletion": 0.20,         # 20% (提高，基于随机性)
            "word_insertion": 0.20,        # 20% (提高，基于随机性)
            "word_replacement": 0.30,      # 30% (提高，基于随机性)
            "word_order_shuffle": True,    # 局部(相邻2-3词)
            "spelling_errors": 0.20,       # 20% (提高，基于随机性)
            "morphological_changes": 0.30, # 30% (提高，基于随机性)
            "abbreviation": 0.60,           # 60% (提高，基于随机性)
            "stopword_removal": 0.70,       # 70% (提高，基于随机性)
        },
        "sentence": {
            "sentence_repetition": 3,            # 降低重复次数（从7降到3），避免文本过长
            "sentence_order_shuffle": True,      # 局部/全局打乱
            "sentence_insertion": 0.40,          # 降低插入数量（从0.80降到0.40），避免文本过长
            "voice_conversion": 1.0,             # 全量应用
            "negation_perturbation": 0.80,       # 进一步增强
            "sentence_truncation": 0.2,          # 保持可用
            "punctuation_perturbation": 1.0,     # 进一步增强，用于计算操作数量
        },
    },
}

# 字符级扰动触发概率（控制每个方法是否触发，进一步降低强度）
CHAR_METHOD_TRIGGER_PROB = {
    "low": 0.35,
    "medium": 0.45,
    "high": 0.55,
}

# 同形字映射（视觉相似字符，拉丁→西里尔）
HOMOGLYPH_MAP = {
    "a": "а",  # 拉丁 -> 西里尔
    "e": "е",
    "o": "о",
    "p": "р",
    "c": "с",
    "x": "х",
    "y": "у",
    "A": "А",
    "E": "Е",
    "O": "О",
    "P": "Р",
    "C": "С",
    "X": "Х",
    "Y": "У",
}

# 相似字符替换映射（用于字符替换）
SIMILAR_CHAR_MAP = {
    "l": "1",
    "i": "1",
    "o": "0",
    "O": "0",
    "s": "5",
    "S": "5",
    "e": "3",
    "E": "3",
    "a": "@",
    "A": "@",
}

# 停用词列表（用于单词级扰动）
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
}

# 常见拼写错误模式（用于单词级扰动）
SPELLING_ERRORS = {
    "receive": "recieve",
    "separate": "seperate",
    "definitely": "definately",
    "accommodate": "acommodate",
    "necessary": "neccessary",
    "occurrence": "occurrance",
    "independent": "independant",
    "beginning": "begining",
    "environment": "enviroment",
    "argument": "arguement",
}

# 常见缩写映射（用于单词级扰动）
ABBREVIATIONS = {
    # 否定形式
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "mightn't": "might not",
    "mayn't": "may not",
    "shan't": "shall not",
    "needn't": "need not",
    "daren't": "dare not",
    "oughtn't": "ought not",
    
    # Be动词缩写
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "why's": "why is",
    "there's": "there is",
    "here's": "here is",
    "let's": "let us",
    
    # Have动词缩写
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    # 注意：he's, she's, it's 在Be动词部分已定义，这里不重复
    # 注意：I'd, you'd, he'd, she'd, we'd, they'd 在Would部分定义（更常见）
    
    # Would/Should/Could缩写（最常见的用法）
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "we'd": "we would",
    "they'd": "they would",
    "that'd": "that would",
    "what'd": "what would",
    "who'd": "who would",
    "where'd": "where would",
    "how'd": "how would",
    "why'd": "why would",
    "there'd": "there would",
    "here'd": "here would",
    
    # 其他常见缩写
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "that'll": "that will",
    "what'll": "what will",
    "who'll": "who will",
    "where'll": "where will",
    "how'll": "how will",
    "why'll": "why will",
    "there'll": "there will",
    "here'll": "here will",
    
    # 时间相关
    "o'clock": "of the clock",
    "'til": "until",
    "till": "until",
    
    # 其他口语缩写
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "hafta": "have to",
    "oughta": "ought to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "musta": "must have",
    "mighta": "might have",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "c'mon": "come on",
    "y'all": "you all",
    "ma'am": "madam",
    "o'er": "over",
    "e'er": "ever",
    "ne'er": "never",
    "e'en": "even",
}

# 常见插入词（用于单词级扰动）
INSERT_WORDS = [
    # 程度副词
    "really", "very", "quite", "rather", "pretty", "so", "just", "too",
    "extremely", "incredibly", "amazingly", "awfully", "terribly",
    "highly", "deeply", "thoroughly", "utterly", "entirely",
    "somewhat", "fairly", "relatively", "comparatively", "reasonably",
    "slightly", "barely", "hardly", "scarcely", "nearly", "almost",
    
    # 强调/肯定词
    "actually", "basically", "essentially", "simply", "merely", "literally",
    "totally", "completely", "absolutely", "definitely", "certainly",
    "surely", "undoubtedly", "obviously", "clearly", "evidently",
    "indeed", "truly", "genuinely", "honestly", "frankly",
    
    # 语气词/填充词
    "well", "um", "uh", "er", "ah", "oh", "hmm", "like",
    "you know", "I mean", "sort of", "kind of", "more or less",
    "or something", "or whatever", "or so", "and stuff", "and things",
    
    # 时间/频率副词
    "always", "often", "usually", "sometimes", "occasionally", "rarely", "never",
    "frequently", "constantly", "regularly", "typically", "normally",
    "already", "still", "yet", "again", "once", "twice", "recently",
    "lately", "nowadays", "currently", "presently",
    
    # 方式副词
    "quickly", "slowly", "carefully", "easily", "hardly", "suddenly",
    "gradually", "rapidly", "instantly", "eventually",
    "finally", "ultimately", "initially", "originally",
    
    # 位置/方向副词
    "here", "there", "everywhere", "anywhere", "somewhere", "nowhere",
    "up", "down", "out", "in", "away", "back", "forward", "ahead",
    
    # 其他常见口语插入词
    "probably", "maybe", "perhaps", "possibly", "likely", "unlikely",
    "apparently", "supposedly", "presumably", "allegedly",
    "especially", "particularly", "specifically", "generally",
    "mostly", "mainly", "primarily", "chiefly", "largely",
    "approximately", "roughly", "about", "around",
    "exactly", "precisely", "accurately", "correctly", "properly",
    "wrongly", "incorrectly", "mistakenly", "unfortunately", "fortunately",
    "hopefully", "thankfully", "luckily", "unluckily", "sadly",
    "surprisingly", "unexpectedly", "naturally",
    "commonly",
    "personally", "individually", "collectively", "together", "separately",
    "similarly", "likewise", "differently", "otherwise", "alternatively",
    "additionally", "furthermore", "moreover", "besides", "also",
    "however", "nevertheless", "nonetheless", "though", "although",
    "therefore", "thus", "hence", "consequently", "accordingly",
    "meanwhile", "simultaneously", "previously", "afterwards",
]

# 无关句子模板（用于句子级扰动）
IRRELEVANT_SENTENCES = [
    # 天气/环境
    "The weather is nice today.",
    "The sun is shining.",
    "It's a beautiful day outside.",
    "The sky is clear and blue.",
    "It's raining outside.",
    "The temperature is perfect today.",
    "There's a gentle breeze blowing.",
    "The clouds are moving slowly.",
    
    # 日常活动
    "I had breakfast this morning.",
    "I'm going to the store later.",
    "I need to do some shopping.",
    "I finished my work early today.",
    "I'm planning to go for a walk.",
    "I'll call you later.",
    "I'm meeting a friend for lunch.",
    "I have a meeting at three o'clock.",
    
    # 物品/位置
    "The book is on the table.",
    "My keys are in my pocket.",
    "The car is parked outside.",
    "I left my phone at home.",
    "The door is open.",
    "There's a pen on the desk.",
    "The window is closed.",
    "I found my wallet.",
    
    # 时间/日期
    "Time passes quickly.",
    "It's already afternoon.",
    "The weekend is coming soon.",
    "Today is Monday.",
    "It's been a long day.",
    "The clock shows three o'clock.",
    "I woke up early this morning.",
    "It's getting late.",
    
    # 音乐/艺术
    "Music is beautiful.",
    "I enjoy listening to songs.",
    "The concert was amazing.",
    "I love playing the piano.",
    "Art can be very inspiring.",
    "The movie was interesting.",
    "I read a good book recently.",
    "The painting looks beautiful.",
    
    # 食物/饮料
    "I like coffee.",
    "The food tastes delicious.",
    "I'm cooking dinner tonight.",
    "Pizza is my favorite food.",
    "I prefer tea over coffee.",
    "The restaurant was crowded.",
    "I'm trying a new recipe.",
    "Breakfast is the most important meal.",
    
    # 技术/科技
    "Technology advances rapidly.",
    "Computers make life easier.",
    "The internet is very useful.",
    "I use my phone every day.",
    "Social media connects people.",
    "The new app is helpful.",
    "I'm learning to code.",
    "Technology changes constantly.",
    
    # 自然/动物
    "Birds are singing in the trees.",
    "The flowers are blooming.",
    "I saw a cat in the garden.",
    "The ocean is vast and deep.",
    "Mountains are majestic and tall.",
    "Forests are full of life.",
    "The river flows gently.",
    "Stars twinkle in the night sky.",
    
    # 运动/健康（非心理健康）
    "Exercise is good for health.",
    "I went for a run today.",
    "Swimming is great exercise.",
    "I enjoy playing basketball.",
    "Walking is good for you.",
    "I'm trying to stay active.",
    "Sports can be fun.",
    "Physical activity is important.",
    
    # 学习/工作
    "I'm studying for an exam.",
    "The class was interesting.",
    "I learned something new today.",
    "Education is valuable.",
    "I have a lot of work to do.",
    "The project is almost finished.",
    "I'm working on a new task.",
    "Learning never stops.",
    
    # 交通/旅行
    "I'm taking the bus today.",
    "The train arrived on time.",
    "I'm planning a vacation.",
    "Traveling is exciting.",
    "I visited a new city.",
    "The flight was smooth.",
    "I love exploring new places.",
    "Road trips can be fun.",
    
    # 家庭/朋友
    "I called my family yesterday.",
    "Friends are important in life.",
    "I'm visiting relatives this weekend.",
    "Family time is precious.",
    "I had dinner with friends.",
    "I'm helping a neighbor.",
    "Community matters a lot.",
    "I'm spending time with loved ones.",
    
    # 购物/消费
    "I bought some groceries.",
    "The store was busy today.",
    "I'm looking for a gift.",
    "Shopping can be enjoyable.",
    "I found a good deal.",
    "The price was reasonable.",
    "I'm saving money.",
    "I need to pay some bills.",
    
    # 兴趣爱好
    "I enjoy reading books.",
    "Photography is my hobby.",
    "I like watching movies.",
    "Cooking is relaxing for me.",
    "I'm learning a new language.",
    "Gardening is peaceful.",
    "I collect stamps.",
    "I love playing games.",
    
    # 一般观察/评论
    "Life is full of surprises.",
    "Every day is different.",
    "Change is constant.",
    "The world is interesting.",
    "There's always something new to discover.",
    "I appreciate simple things.",
    "Small moments matter.",
    "I'm grateful for today.",
]


# ============================
# 字符级扰动实现（完全使用正则表达式和Python字符串操作）
# ============================

def apply_character_perturbation(
    text: str,
    intensity: Literal["low", "medium", "high"] = "medium",
) -> str:
    """应用字符级扰动。

    实现字符级的全部扰动方法（严格按照扰动方法.md）：
    1. 字符删除（随机删除字符）
    2. 字符插入（随机位置插入字符）
    3. 字符替换（替换为视觉相似字符）
    4. 字符交换（相邻字符互换）
    5. 大小写变换
    6. 同形字替换
    7. 重复字符

    Args:
        text: 待扰动的文本
        intensity: 扰动强度（low/medium/high）

    Returns:
        扰动后的文本
    """
    if not text:
        return text

    rates = PERTURBATION_INTENSITY[intensity]["character"]
    result = text

    # 为每个方法设置随机触发开关，进一步降低总体扰动强度
    trigger_prob = CHAR_METHOD_TRIGGER_PROB[intensity]

    # 1. 字符删除（随机删除字符）
    if random.random() < trigger_prob:
        result = _apply_character_deletion(result, rates["deletion"])

    # 2. 字符插入（随机位置插入字符）
    if random.random() < trigger_prob:
        result = _apply_character_insertion(result, rates["insertion"])

    # 3. 字符替换（替换为视觉相似字符）
    if random.random() < trigger_prob:
        result = _apply_character_replacement(result, rates["replacement"])

    # 4. 字符交换（相邻字符互换）
    if random.random() < trigger_prob:
        result = _apply_character_swap(result, rates["swap"])

    # 5. 大小写变换
    if random.random() < trigger_prob:
        result = _apply_case_change(result, rates["case_change"])

    # 6. 同形字替换
    if random.random() < trigger_prob:
        result = _apply_homoglyph_replacement(result, rates["homoglyph"])

    # 7. 重复字符（字符重复输入，1-3次重复）
    if random.random() < trigger_prob:
        result = _apply_character_repetition(result, rates["repetition"])

    return result


def _apply_character_deletion(text: str, rate: float) -> str:
    """随机删除字符（基于概率的随机性）。"""
    if not text or rate <= 0:
        return text
    chars = list(text)
    result = []
    for char in chars:
        if random.random() >= rate:  # 以(1-rate)的概率保留字符
            result.append(char)
    return "".join(result) if result else text


def _apply_character_insertion(text: str, rate: float) -> str:
    """随机位置插入字符（基于概率的随机性）。"""
    if not text or rate <= 0:
        return text
    chars = list(text)
    result = []
    for char in chars:
        result.append(char)
        if random.random() < rate:  # 以rate概率在字符后插入
            insert_char = random.choice(string.ascii_letters + string.digits)
            result.append(insert_char)
    return "".join(result)


def _apply_character_replacement(text: str, rate: float) -> str:
    """随机替换为相似字符（基于概率的随机性，使用SIMILAR_CHAR_MAP）。"""
    if not text or rate <= 0:
        return text
    chars = list(text)
    result = []
    for char in chars:
        if random.random() < rate:  # 以rate概率替换字符
            if char in SIMILAR_CHAR_MAP:
                result.append(SIMILAR_CHAR_MAP[char])
            elif char.isalpha():
                result.append(random.choice(string.ascii_letters))
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def _apply_character_swap(text: str, rate: float) -> str:
    """相邻字符位置互换（基于概率的随机性）。"""
    if not text or rate <= 0 or len(text) < 2:
        return text
    chars = list(text)
    result = chars.copy()
    # 遍历相邻字符对，以rate概率交换
    for i in range(len(result) - 1):
        if random.random() < rate:
            result[i], result[i + 1] = result[i + 1], result[i]
    return "".join(result)


def _apply_case_change(text: str, rate: float) -> str:
    """随机改变大小写（基于概率的随机性）。"""
    if not text or rate <= 0:
        return text
    chars = list(text)
    result = []
    for char in chars:
        if char.isalpha() and random.random() < rate:  # 以rate概率改变大小写
            result.append(char.swapcase())
        else:
            result.append(char)
    return "".join(result)


def _apply_homoglyph_replacement(text: str, rate: float) -> str:
    """使用视觉相似字符替换（基于概率的随机性，使用HOMOGLYPH_MAP）。"""
    if not text or rate <= 0:
        return text
    chars = list(text)
    result = []
    for char in chars:
        if char in HOMOGLYPH_MAP and random.random() < rate:  # 以rate概率替换
            result.append(HOMOGLYPH_MAP[char])
        else:
            result.append(char)
    return "".join(result)


def _apply_character_repetition(text: str, repetition_prob: float) -> str:
    """字符重复输入（1-3次重复）。"""
    if not text:
        return text
    chars = list(text)
    result = []
    for char in chars:
        result.append(char)
        if random.random() < repetition_prob:
            repeat_count = random.randint(1, 3)
            result.append(char * repeat_count)
    return "".join(result)


# ============================
# 单词级扰动实现（基于规则的实现）
# ============================

def _random_bool(probability: float) -> bool:
    """返回True的概率为probability。

    Args:
        probability: 返回True的概率，区间[0, 1]。

    Returns:
        以probability概率返回True。
    """
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return random.random() < probability


def apply_word_perturbation(
    text: str,
    intensity: Literal["low", "medium", "high"] = "medium",
) -> str:
    """应用单词级扰动（基于规则的实现）。

    实现单词级的全部扰动方法（严格按照扰动方法.md）：
    1. 单词删除
    2. 单词插入（插入无关词）
    3. 单词替换（随机替换）
    4. 单词顺序打乱（局部相邻2-3词）
    5. 拼写错误
    6. 词形变化（简化实现）
    7. 缩写扩展/压缩
    8. 停用词删除

    Args:
        text: 待扰动的文本
        intensity: 扰动强度（low/medium/high）

    Returns:
        扰动后的文本
    """
    if not text:
        return text

    rates = PERTURBATION_INTENSITY[intensity]["word"]
    result = text

    # 1. 单词删除
    result = _apply_word_deletion(result, rates["word_deletion"])

    # 2. 单词插入
    result = _apply_word_insertion(result, rates["word_insertion"])

    # 3. 单词替换（随机替换）
    result = _apply_word_replacement(result, rates["word_replacement"])

    # 4. 单词顺序打乱（局部相邻2-3词）
    result = _apply_word_order_shuffle(result, max_range=2)

    # 5. 拼写错误
    result = _apply_spelling_errors(result, rates["spelling_errors"])

    # 6. 词形变化（简化实现：随机删除词尾）
    result = _apply_morphological_changes(result, rates["morphological_changes"])

    # 7. 缩写扩展/压缩
    result = _apply_abbreviation_perturbation(result, rates["abbreviation"])

    # 8. 停用词删除
    result = _apply_stopword_removal(result, rates["stopword_removal"])

    return result


def _apply_word_deletion(text: str, rate: float) -> str:
    """随机删除单词。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    result = [word for word in words if not _random_bool(rate)]
    return " ".join(result) if result else text


def _apply_word_insertion(text: str, rate: float) -> str:
    """随机插入无关单词。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    result = []
    for word in words:
        result.append(word)
        if _random_bool(rate):
            insert_word = random.choice(INSERT_WORDS)
            result.append(insert_word)
    return " ".join(result)


def _apply_word_replacement(text: str, rate: float) -> str:
    """随机替换单词（简化实现：随机替换为相似长度的随机词）。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    result = []
    for word in words:
        if _random_bool(rate) and word.isalpha() and len(word) > 2:
            # 简化的替换：随机生成相似长度的词
            replacement = "".join(random.choices(string.ascii_lowercase, k=len(word)))
            result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


def _apply_word_order_shuffle(text: str, max_range: int = 2) -> str:
    """单词顺序打乱（局部相邻2-3词）。"""
    if not text:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    # 使用类似random_token_permutation的方法
    new_indices = [i + random.uniform(0, max_range + 1) for i in range(len(words))]
    result = [x for _, x in sorted(zip(new_indices, words), key=lambda pair: pair[0])]
    return " ".join(result)


def _apply_spelling_errors(text: str, rate: float) -> str:
    """添加常见拼写错误。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    result = []
    for word in words:
        word_lower = word.lower()
        if word_lower in SPELLING_ERRORS and _random_bool(rate):
            # 保持原始大小写
            if word[0].isupper():
                result.append(SPELLING_ERRORS[word_lower].capitalize())
            else:
                result.append(SPELLING_ERRORS[word_lower])
        else:
            result.append(word)
    return " ".join(result)


def _apply_morphological_changes(text: str, rate: float) -> str:
    """词形变化（简化实现：随机删除常见词尾）。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    suffixes = ["ing", "ed", "er", "est", "ly", "s", "es"]
    result = []
    for word in words:
        if _random_bool(rate) and word.isalpha() and len(word) > 4:
            # 尝试删除常见词尾
            for suffix in suffixes:
                if word.lower().endswith(suffix) and len(word) > len(suffix) + 1:
                    new_word = word[:-len(suffix)]
                    result.append(new_word)
                    break
            else:
                result.append(word)
        else:
            result.append(word)
    return " ".join(result)


def _apply_abbreviation_perturbation(text: str, rate: float) -> str:
    """缩写扩展/压缩。"""
    if not text or rate <= 0:
        return text
    result = text
    # 尝试扩展缩写
    for abbrev, expanded in ABBREVIATIONS.items():
        if abbrev in result and _random_bool(rate):
            result = result.replace(abbrev, expanded)
    # 尝试压缩（反向）
    for abbrev, expanded in ABBREVIATIONS.items():
        if expanded in result and _random_bool(rate):
            result = result.replace(expanded, abbrev)
    return result


def _apply_stopword_removal(text: str, rate: float) -> str:
    """删除停用词。"""
    if not text or rate <= 0:
        return text
    words = text.split()
    if not words:
        return text
    result = [word for word in words if word.lower() not in STOP_WORDS or not _random_bool(rate)]
    return " ".join(result) if result else text


# ============================
# 句子级扰动实现（基于规则的实现）
# ============================

def apply_sentence_perturbation(
    inputs: List[str],
    intensity: Literal["low", "medium", "high"] = "medium",
) -> List[str]:
    """应用句子级扰动（基于规则的实现）。

    实现句子级的全部扰动方法（严格按照扰动方法.md）：
    1. 句子重复
    2. 句子顺序打乱
    3. 句子插入（插入无关句子）
    4. 句式转换（主动/被动，简化实现）
    5. 否定词添加/删除
    6. 句子截断
    7. 标点符号扰动

    Args:
        inputs: 待扰动的句子列表
        intensity: 扰动强度（low/medium/high）

    Returns:
        扰动后的句子列表
    """
    if not inputs:
        return inputs

    rates = PERTURBATION_INTENSITY[intensity]["sentence"]
    result = inputs.copy()

    # 1. 句子重复
    result = _apply_sentence_repetition(result, rates["sentence_repetition"])

    # 2. 句子顺序打乱
    result = _apply_sentence_order_shuffle(result)

    # 3. 句子插入（插入无关句子）
    result = _apply_sentence_insertion(result, rates["sentence_insertion"])

    # 4. 句式转换（简化实现：添加/删除被动语态标记）
    result = _apply_voice_conversion(result, rates["voice_conversion"])

    # 5. 否定词添加/删除
    result = _apply_negation_perturbation(result, rates["negation_perturbation"])

    # 6. 句子截断
    if len(result) > 2:
        result = _apply_sentence_truncation(result)

    # 7. 标点符号扰动
    result = _apply_punctuation_perturbation(result, rates["punctuation_perturbation"])

    return result


def _apply_sentence_repetition(sentences: List[str], max_repetitions: int) -> List[str]:
    """Sentence repetition (apply to all sentences, no randomness, increased intensity)."""
    if not sentences:
        return sentences
    result = []
    # Increase intensity: use max_repetitions + 1 to ensure stronger perturbation
    repeat_count = max_repetitions + 1
    for sentence in sentences:
        result.append(sentence)
        # Apply repetition to all sentences with increased count
        result.extend([sentence] * repeat_count)
    return result


def _apply_sentence_order_shuffle(sentences: List[str]) -> List[str]:
    """句子顺序打乱：以标点符号为分割，交换相邻句子。
    
    使用正则表达式按标点符号（句号、问号、感叹号等）分割句子，
    然后随机交换相邻的句子对。
    
    Args:
        sentences: 句子列表
        
    Returns:
        交换后的句子列表
    """
    if not sentences:
        return sentences
    
    result = []
    # 定义句子结束标点符号的正则表达式（中英文标点）
    # 匹配：非标点字符 + 标点符号（一个或多个），用于分割
    sentence_end_pattern = r'[.!?。！？]+'
    
    for sentence in sentences:
        if not sentence or len(sentence.strip()) == 0:
            result.append(sentence)
            continue
        
        # 使用正则表达式分割句子，保留分隔符（标点符号）
        # 使用捕获组来保留分隔符
        parts = re.split(f'({sentence_end_pattern})', sentence)
        
        # 过滤空字符串
        parts = [p for p in parts if p]
        
        if len(parts) < 3:  # 至少需要：内容+标点+内容 才能形成两个句子
            result.append(sentence)
            continue
        
        # 将分割后的部分组合成句子（内容+标点）
        sentence_list = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(sentence_end_pattern, parts[i + 1]):
                # 如果下一个部分是标点，合并当前内容和标点
                sentence_list.append(parts[i] + parts[i + 1])
                i += 2
            else:
                # 单独的内容部分（可能是文本末尾没有标点的部分）
                if parts[i].strip():  # 只添加非空内容
                    sentence_list.append(parts[i])
                i += 1
        
        # 如果只有一个句子，无法交换
        if len(sentence_list) < 2:
            result.append(sentence)
            continue
        
        # 交换相邻的句子对（随机选择一些相邻对进行交换）
        swapped_list = sentence_list.copy()
        # 遍历相邻句子对，随机决定是否交换
        i = 0
        while i < len(swapped_list) - 1:
            if random.random() < 0.5:  # 50%概率交换相邻句子对
                swapped_list[i], swapped_list[i + 1] = swapped_list[i + 1], swapped_list[i]
                i += 2  # 跳过已交换的句子，避免重复交换
            else:
                i += 1
        
        # 重新组合成完整文本
        result.append(''.join(swapped_list))
    
    return result


def _apply_sentence_insertion(sentences: List[str], rate: float) -> List[str]:
    """Insert irrelevant sentences (apply to all sentences, no randomness, increased intensity)."""
    if not sentences or rate <= 0:
        return sentences
    result = []
    # Increase intensity: insert multiple irrelevant sentences per original sentence
    insertion_count = max(1, int(rate * 10))  # Convert rate to count (e.g., 0.1 -> 1, 0.2 -> 2)
    for sentence in sentences:
        result.append(sentence)
        # Apply insertion to all sentences with increased count
        for _ in range(insertion_count):
            irrelevant_sentence = random.choice(IRRELEVANT_SENTENCES)
            result.append(irrelevant_sentence)
    return result


def _apply_voice_conversion(sentences: List[str], rate: float) -> List[str]:
    """Voice conversion (apply to all sentences, no randomness)."""
    if not sentences or rate <= 0:
        return sentences
    result = []
    passive_markers = ["by", "was", "were", "is", "are", "been", "being"]
    for sentence in sentences:
        # Apply to all sentences
        words = sentence.split()
        if not words:
            result.append(sentence)
            continue
        
        # Try to convert between active and passive voice
        # Simplified: add or remove passive markers
        has_passive = any(marker in sentence.lower() for marker in passive_markers)
        
        if has_passive:
            # Remove passive markers (simplified)
            for marker in passive_markers:
                if marker in sentence.lower():
                    # Remove the marker (case-insensitive)
                    import re
                    sentence = re.sub(re.escape(marker), "", sentence, flags=re.IGNORECASE)
                    sentence = " ".join(sentence.split())  # Normalize whitespace
                    break
        else:
            # Add passive markers
            if len(words) > 2:
                # Add "by" before the last word (simplified passive construction)
                insert_pos = max(1, len(words) - 2)
                words.insert(insert_pos, "by")
                sentence = " ".join(words)
        result.append(sentence)
    return result


def _apply_negation_perturbation(sentences: List[str], rate: float) -> List[str]:
    """Add/remove negation words (apply to all sentences, no randomness)."""
    if not sentences or rate <= 0:
        return sentences
    negation_words = ["not", "no", "never", "nothing", "nobody", "nowhere", "none", "neither", "nor"]
    result = []
    for sentence in sentences:
        # Apply to all sentences
        # Check if sentence already has negation
        has_negation = any(neg in sentence.lower() for neg in negation_words)
        if has_negation:
            # Remove negation word (simplified)
            for neg in negation_words:
                if neg in sentence.lower():
                    import re
                    sentence = re.sub(r'\b' + re.escape(neg) + r'\b', "", sentence, flags=re.IGNORECASE)
                    sentence = " ".join(sentence.split())  # Normalize whitespace
                    break
        else:
            # Add negation word
            neg_word = random.choice(negation_words)
            words = sentence.split()
            if words:
                insert_pos = min(1, len(words) - 1)  # Fixed position instead of random
                words.insert(insert_pos, neg_word)
                sentence = " ".join(words)
        result.append(sentence)
    return result


def _apply_sentence_truncation(sentences: List[str]) -> List[str]:
    """Sentence truncation (apply to all sentences, no randomness)."""
    if not sentences:
        return sentences
    result = []
    # Apply truncation to all sentences (use "rear" type as default)
    truncation_type = "rear"
    for sentence in sentences:
        # Apply truncation to all sentences
        if truncation_type == "front" and len(sentence) > 4:
            # Keep last 75%
            result.append(sentence[int(len(sentence) * 0.25):])
        elif truncation_type == "middle" and len(sentence) > 4:
            # Keep first 25% and last 25%
            front_len = int(len(sentence) * 0.25)
            rear_len = int(len(sentence) * 0.25)
            result.append(sentence[:front_len] + "..." + sentence[-rear_len:])
        elif truncation_type == "rear" and len(sentence) > 4:
            # Keep first 75%
            result.append(sentence[:int(len(sentence) * 0.75)] + "...")
        else:
            result.append(sentence)
    return result


def _apply_punctuation_perturbation(sentences: List[str], rate: float) -> List[str]:
    """Punctuation perturbation (apply to all sentences, no randomness, increased intensity)."""
    if not sentences or rate <= 0:
        return sentences
    punctuation = [".", ",", "!", "?", ";", ":", "-", "'", '"']
    result = []
    # Increase intensity: remove multiple punctuations and add multiple
    removal_count = max(1, int(rate * 5))  # Remove more punctuations
    insertion_count = max(1, int(rate * 5))  # Add more punctuations
    for sentence in sentences:
        # Apply to all sentences with increased intensity
        # Remove multiple punctuations
        removed = 0
        for punct in punctuation:
            while punct in sentence and removed < removal_count:
                sentence = sentence.replace(punct, "", 1)
                removed += 1
                if removed >= removal_count:
                    break
            if removed >= removal_count:
                break
        
        # Add multiple punctuations to all sentences
        if len(sentence) > 0:
            for i in range(insertion_count):
                # Distribute insertions evenly across the sentence
                insert_pos = int(len(sentence) * (i + 1) / (insertion_count + 1))
                if insert_pos < len(sentence):
                    punct = punctuation[i % len(punctuation)]  # Cycle through punctuation
                    sentence = sentence[:insert_pos] + punct + sentence[insert_pos:]
        result.append(sentence)
    return result


# ============================
# 综合扰动接口
# ============================

def apply_perturbation(
    data: Any,
    intensity: Literal["low", "medium", "high"] = "medium",
    perturbation_type: Optional[Literal["character", "word", "sentence", "all"]] = "all",
) -> Any:
    """统一的综合扰动接口（自动应用字符/单词/句子三级扰动）。

    为了保持兼容性，本函数完全基于现有三个扰动实现进行组合，不改变其内部逻辑。

    当前策略：
        1. 对每个文本先应用字符级扰动
        2. 再应用单词级扰动
        3. 最后对整个句子列表应用句子级扰动

    Args:
        data: 待扰动的数据
              - 通常为字符串列表（多轮对话）
              - 也兼容单个字符串
        intensity: 扰动强度（low/medium/high）
        perturbation_type: 扰动类型（character/word/sentence/all）

    Returns:
        扰动后的数据，类型与输入保持一致。
    """
    # 列表输入（典型：对话轮次列表）
    if isinstance(data, list):
        tmp = data.copy()
        # 根据扰动类型应用不同的扰动
        if perturbation_type in ["character", "all"]:
            tmp = [apply_character_perturbation(str(t), intensity) for t in tmp]
        if perturbation_type in ["word", "all"]:
            tmp = [apply_word_perturbation(t, intensity) for t in tmp]
        if perturbation_type in ["sentence", "all"]:
            tmp = apply_sentence_perturbation(tmp, intensity)
        return tmp

    # 单个字符串输入
    text = str(data)
    if perturbation_type in ["character", "all"]:
        text = apply_character_perturbation(text, intensity)
    if perturbation_type in ["word", "all"]:
        text = apply_word_perturbation(text, intensity)
    # 句子级扰动在单文本场景下意义有限，这里不额外包装列表，直接返回
    return text


def perturb_samples(
    samples: List[Dict[str, Any]],
    intensity: Literal["low", "medium", "high"] = "medium",
    perturbation_type: Optional[Literal["character", "word", "sentence", "all"]] = "all",
) -> List[Dict[str, Any]]:
    """对分类输入进行综合扰动（字符/单词/句子三级组合）。

    为与 `LLMs-Mental-Health-Crisis` 保持一致，这里不再区分扰动类型，
    而是对每条样本的 `inputs` 依次施加字符级、单词级和句子级扰动。

    Args:
        samples: 样本列表
        intensity: 扰动强度（low/medium/high）
        perturbation_type: 扰动类型（character/word/sentence/all）

    Returns:
        扰动后的样本列表
    """
    perturbed_samples: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc=f"应用综合扰动[强度: {intensity}, 类型: {perturbation_type}]",
        unit="样本",
        leave=False,
        total=len(samples),
    )

    for item in iterator:
        perturbed_item = dict(item)
        inputs = item.get("inputs", [])
        if not isinstance(inputs, list):
            inputs = [inputs]

        # 使用统一的综合扰动接口
        perturbed_item["inputs"] = apply_perturbation(inputs, intensity, perturbation_type)
        perturbed_samples.append(perturbed_item)

    return perturbed_samples


def save_perturbed_data(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """保存扰动后的数据到CSV文件（保持原有CSV格式，只包含text和label两列）。

    Args:
        samples: 扰动后的样本列表
        output_path: 输出文件路径（CSV格式）
    """
    import pandas as pd
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备CSV数据：只包含text和label两列，保持与原始数据集格式一致
    data_rows = []
    for sample in samples:
        # 获取扰动后的文本（从inputs中提取）
        inputs = sample.get("inputs", [])
        if isinstance(inputs, list) and len(inputs) > 0:
            # 如果inputs是列表，取第一个元素（通常是文本）
            perturbed_text = str(inputs[0]) if inputs[0] else sample.get("text", "")
        else:
            perturbed_text = str(inputs) if inputs else sample.get("text", "")
        
        # 获取原始标签格式（保持原有格式，如 self.Anxiety）
        raw_label = sample.get("raw_label", sample.get("label", ""))
        
        data_rows.append({
            "text": perturbed_text,
            "label": raw_label,
        })
    
    # 创建DataFrame并保存为CSV（使用与原始数据相同的格式）
    df = pd.DataFrame(data_rows)
    # 使用csv.QUOTE_MINIMAL，与原始CSV格式保持一致
    df.to_csv(
        output_path, 
        index=False, 
        encoding="utf-8", 
        quoting=csv.QUOTE_MINIMAL,
        escapechar=None,
        doublequote=True,
    )
    print(f"扰动后的数据已保存至: {output_path} (CSV格式, {len(data_rows)} 条样本)")
