
# 第9章：创意生成 Agent 开发实践

创意生成 Agent 是一种能够在各种领域产生新颖、有价值想法的 AI 系统。这种 Agent 可以应用于广告创意、产品设计、故事创作等多个领域。本章将探讨如何开发这样一个系统。

## 9.1 创意生成系统设计

### 9.1.1 创意领域定义

首先，我们需要明确定义创意生成的领域和范围。

示例代码：

```python
from enum import Enum
from typing import List, Dict, Any

class CreativeDomain(Enum):
    ADVERTISING = "advertising"
    PRODUCT_DESIGN = "product_design"
    STORY_WRITING = "story_writing"
    MUSIC_COMPOSITION = "music_composition"

class CreativeTask:
    def __init__(self, domain: CreativeDomain, description: str, constraints: List[str] = None):
        self.domain = domain
        self.description = description
        self.constraints = constraints or []

class CreativeDomainManager:
    def __init__(self):
        self.domains = {}

    def register_domain(self, domain: CreativeDomain, attributes: Dict[str, Any]):
        self.domains[domain] = attributes

    def get_domain_attributes(self, domain: CreativeDomain) -> Dict[str, Any]:
        return self.domains.get(domain, {})

# 使用示例
domain_manager = CreativeDomainManager()

domain_manager.register_domain(
    CreativeDomain.ADVERTISING,
    {
        "target_audience": ["young_adults", "professionals", "seniors"],
        "media_types": ["print", "digital", "video"],
        "tone": ["humorous", "serious", "inspirational"]
    }
)

domain_manager.register_domain(
    CreativeDomain.STORY_WRITING,
    {
        "genres": ["science_fiction", "romance", "mystery", "fantasy"],
        "length": ["short_story", "novella", "novel"],
        "viewpoint": ["first_person", "third_person", "omniscient"]
    }
)

# 创建一个创意任务
ad_task = CreativeTask(
    CreativeDomain.ADVERTISING,
    "Create a digital ad campaign for a new smartphone",
    ["target millennials", "emphasize AI features", "use vibrant colors"]
)

# 获取领域属性
ad_attributes = domain_manager.get_domain_attributes(CreativeDomain.ADVERTISING)
print(f"Advertising domain attributes: {ad_attributes}")
```

### 9.1.2 生成流程设计

接下来，我们需要设计创意生成的流程，包括灵感收集、创意生成、评估和优化等步骤。

示例代码：

```python
from abc import ABC, abstractmethod

class CreativeStage(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class InspirationGathering(CreativeStage):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 实现灵感收集逻辑
        return {"inspirations": ["AI assistant", "Futuristic UI", "Eco-friendly tech"]}

class IdeaGeneration(CreativeStage):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 实现创意生成逻辑
        return {"ideas": ["AI-powered personal assistant", "Holographic display", "Solar-charged battery"]}

class CreativeEvaluation(CreativeStage):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 实现创意评估逻辑
        return {"evaluations": [{"idea": "AI-powered personal assistant", "score": 0.8}]}

class CreativeRefinement(CreativeStage):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 实现创意优化逻辑
        return {"refined_ideas": ["Advanced AI assistant with predictive capabilities"]}

class CreativeWorkflow:
    def __init__(self):
        self.stages = []

    def add_stage(self, stage: CreativeStage):
        self.stages.append(stage)

    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        result = initial_input
        for stage in self.stages:
            result = stage.process(result)
        return result

# 使用示例
workflow = CreativeWorkflow()
workflow.add_stage(InspirationGathering())
workflow.add_stage(IdeaGeneration())
workflow.add_stage(CreativeEvaluation())
workflow.add_stage(CreativeRefinement())

initial_input = {"task": ad_task}
final_result = workflow.execute(initial_input)
print(f"Final creative output: {final_result}")
```

### 9.1.3 评估指标确立

建立适当的评估指标对于衡量和改进创意的质量至关重要。

示例代码：

```python
from typing import List

class CreativeMetric:
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def evaluate(self, idea: str) -> float:
        # 这里应该实现具体的评估逻辑
        # 为简化示例，我们返回一个随机分数
        return random.random()

class CreativeEvaluator:
    def __init__(self, metrics: List[CreativeMetric]):
        self.metrics = metrics

    def evaluate_idea(self, idea: str) -> Dict[str, float]:
        scores = {}
        for metric in self.metrics:
            scores[metric.name] = metric.evaluate(idea)
        
        overall_score = sum(metric.weight * scores[metric.name] for metric in self.metrics)
        scores["overall"] = overall_score

        return scores

# 使用示例
metrics = [
    CreativeMetric("novelty", 0.3),
    CreativeMetric("relevance", 0.3),
    CreativeMetric("feasibility", 0.2),
    CreativeMetric("impact", 0.2)
]

evaluator = CreativeEvaluator(metrics)

idea = "AI-powered personal assistant with emotion recognition"
scores = evaluator.evaluate_idea(idea)

print(f"Evaluation scores for '{idea}':")
for metric, score in scores.items():
    print(f"  {metric}: {score:.2f}")
```

这个设计为创意生成 Agent 提供了一个灵活的框架：

1. `CreativeDomainManager` 允许定义和管理不同的创意领域。
2. `CreativeTask` 封装了特定的创意任务及其约束。
3. `CreativeWorkflow` 定义了一个可定制的创意生成流程。
4. `CreativeEvaluator` 提供了一种评估创意质量的方法。

在实际应用中，你可能需要：

1. 实现更复杂的灵感收集算法，如网络爬虫或知识图谱查询。
2. 使用高级的自然语言处理或机器学习模型来生成创意。
3. 开发更精细的评估指标，可能涉及人工智能或众包评估。
4. 添加用户反馈循环，以持续改进创意生成过程。
5. 实现跨领域创意生成，结合不同领域的特点产生新颖想法。

## 9.2 灵感源与知识库构建

灵感源和知识库是创意生成的基础，它们为 Agent 提供了产生新想法所需的原材料。

### 9.2.1 多源数据采集

从多个来源收集相关数据可以丰富创意生成的基础。

示例代码：

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class DataSource(ABC):
    @abstractmethod
    def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        pass

class WebScraper(DataSource):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/search?q={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for item in soup.find_all('div', class_='search-result'):
            results.append({
                'title': item.find('h2').text,
                'description': item.find('p').text,
                'url': item.find('a')['href']
            })
        return results

class APIClient(DataSource):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(f"{self.api_url}?q={query}", headers=headers)
        return response.json()['results']

class DataCollector:
    def __init__(self):
        self.sources = []

    def add_source(self, source: DataSource):
        self.sources.append(source)

    def collect_data(self, query: str) -> List[Dict[str, Any]]:
        all_data = []
        for source in self.sources:
            all_data.extend(source.fetch_data(query))
        return all_data

# 使用示例
web_scraper = WebScraper("https://example.com")
api_client = APIClient("https://api.example.com/v1/search", "your_api_key_here")

collector = DataCollector()
collector.add_source(web_scraper)
collector.add_source(api_client)

data = collector.collect_data("AI in smartphone design")
print(f"Collected {len(data)} items")
```

### 9.2.2 创意元素提取

从收集的数据中提取关键的创意元素。

示例代码：

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

class CreativeElementExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return [word for word, _ in Counter(words).most_common(top_n)]

    def extract_phrases(self, text: str, top_n: int = 5) -> List[str]:
        words = word_tokenize(text.lower())
        bigrams = nltk.bigrams(words)
        bigrams = [' '.join(bg) for bg in bigrams if all(word not in self.stop_words for word in bg)]
        return [phrase for phrase, _ in Counter(bigrams).most_common(top_n)]

    def extract_elements(self, data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        all_text = ' '.join([item['title'] + ' ' + item['description'] for item in data])
        return {
            'keywords': self.extract_keywords(all_text),
            'phrases': self.extract_phrases(all_text)
        }

# 使用示例
extractor = CreativeElementExtractor()
elements = extractor.extract_elements(data)
print("Extracted creative elements:")
print(f"Keywords: {elements['keywords']}")
print(f"Phrases: {elements['phrases']}")
```

### 9.2.3 知识图谱构建

构建知识图谱可以帮助 Agent 理解概念之间的关系，从而产生更有洞察力的创意。

示例代码：

```python
from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_relation(self, entity1: str, relation: str, entity2: str):
        self.graph.add_edge(entity1, entity2, relation=relation)

    def get_related_entities(self, entity: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        related = []
        for node in self.graph.nodes():
            if node != entity:
                try:
                    distance = nx.shortest_path_length(self.graph, entity, node)
                    if distance <= max_distance:
                        related.append((node, distance))
                except nx.NetworkXNoPath:
                    pass
        return sorted(related, key=lambda x: x[1])

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.axis('off')
        plt.show()

# 使用示例
kg = KnowledgeGraph()

# 添加一些关系
kg.add_relation("Smartphone", "has_feature", "AI Assistant")
kg.add_relation("AI Assistant", "uses", "Natural Language Processing")
kg.add_relation("Smartphone", "has_component", "Camera")
kg.add_relation("Camera", "uses", "Image Recognition")
kg.add_relation("Image Recognition", "is_a", "AI Technology")

# 获取相关实体
related = kg.get_related_entities("Smartphone")
print("Entities related to Smartphone:")
for entity, distance in related:
    print(f"  {entity} (distance: {distance})")

# 可视化知识图谱
kg.visualize()
```

这些组件共同工作，可以为创意生成 Agent 提供丰富的灵感源和结构化的知识：

1. `DataCollector` 从多个来源收集相关数据。
2. `CreativeElementExtractor` 从原始数据中提取关键词和短语。
3. `KnowledgeGraph` 构建和管理概念之间的关系网络。

在实际应用中，你可能需要：

1. 实现更复杂的数据清洗和预处理步骤。
2. 使用高级的自然语言处理技术，如命名实体识别或关系提取，来改进创意元素的提取。
3. 集成专业的知识图谱数据库，如 Neo4j，以处理大规模的知识网络。
4. 实现自动化的知识图谱更新机制，以保持知识的时效性。
5. 开发特定领域的本体模型，以更准确地表示领域知识。

通过这些技术，创意生成 Agent 可以获得广泛而深入的知识基础，从而产生更加新颖和有价值的创意。这个知识库不仅可以提供直接的灵感，还可以帮助 Agent 建立意想不到的概念联系，这是创新思维的关键。

## 9.3 LLM 创意生成技术

利用大型语言模型（LLM）可以显著提升创意生成的质量和多样性。以下是一些关键技术：

### 9.3.1 条件生成方法

条件生成允许我们控制 LLM 的输出，使其符合特定的创意要求。

示例代码：

```python
import openai
from typing import List, Dict, Any

class LLMCreativeGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_ideas(self, prompt: str, constraints: List[str], num_ideas: int = 3) -> List[str]:
        formatted_prompt = f"{prompt}\n\nConstraints:\n" + "\n".join([f"- {c}" for c in constraints])
        formatted_prompt += f"\n\nGenerate {num_ideas} creative ideas based on the above prompt and constraints:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=formatted_prompt,
            max_tokens=200,
            n=num_ideas,
            stop=None,
            temperature=0.8,
        )

        return [choice.text.strip() for choice in response.choices]

# 使用示例
generator = LLMCreativeGenerator("your_openai_api_key_here")

prompt = "Design a new smartphone feature that enhances user productivity"
constraints = [
    "Must be AI-powered",
    "Should work offline",
    "Must respect user privacy"
]

ideas = generator.generate_ideas(prompt, constraints)
for i, idea in enumerate(ideas, 1):
    print(f"Idea {i}: {idea}")
```

### 9.3.2 风格迁移技术

风格迁移可以帮助我们将一种创意风格应用到另一个领域或概念上。

示例代码：

```python
class StyleTransferGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_with_style(self, content: str, style: str) -> str:
        prompt = f"Transform the following content into the style of {style}:\n\n{content}\n\nStyled version:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
style_generator = StyleTransferGenerator("your_openai_api_key_here")

content = "A smartphone app that helps users manage their daily tasks"
style = "a cyberpunk novel"

styled_content = style_generator.generate_with_style(content, style)
print(f"Original: {content}")
print(f"Styled: {styled_content}")
```

### 9.3.3 多样性增强策略

为了避免生成重复或相似的创意，我们可以实施多样性增强策略。

示例代码：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DiversityEnhancer:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.vectorizer = TfidfVectorizer()

    def generate_diverse_ideas(self, prompt: str, num_ideas: int = 5, diversity_threshold: float = 0.3) -> List[str]:
        ideas = []
        attempts = 0
        max_attempts = num_ideas * 3

        while len(ideas) < num_ideas and attempts < max_attempts:
            new_idea = self._generate_single_idea(prompt)
            if self._is_diverse(new_idea, ideas, diversity_threshold):
                ideas.append(new_idea)
            attempts += 1

        return ideas

    def _generate_single_idea(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.9,
        )
        return response.choices[0].text.strip()

    def _is_diverse(self, new_idea: str, existing_ideas: List[str], threshold: float) -> bool:
        if not existing_ideas:
            return True

        all_ideas = existing_ideas + [new_idea]
        tfidf_matrix = self.vectorizer.fit_transform(all_ideas)
        cosine_similarities = (tfidf_matrix * tfidf_matrix.T).A

        max_similarity = np.max(cosine_similarities[-1][:-1])
        return max_similarity < threshold

# 使用示例
enhancer = DiversityEnhancer("your_openai_api_key_here")

prompt = "Generate a unique feature for a smart home device:"
diverse_ideas = enhancer.generate_diverse_ideas(prompt)

print("Diverse smart home device features:")
for i, idea in enumerate(diverse_ideas, 1):
    print(f"{i}. {idea}")
```

这些 LLM 创意生成技术为我们提供了强大的工具来产生新颖和多样化的创意：

1. `LLMCreativeGenerator` 使用条件生成方法，根据给定的提示和约束生成创意。
2. `StyleTransferGenerator` 实现了风格迁移，可以将一个创意概念转换为特定的风格。
3. `DiversityEnhancer` 通过检查生成的创意之间的相似性，确保输出的多样性。

在实际应用中，你可能需要：

1. 实现更复杂的提示工程技术，以更好地引导 LLM 的输出。
2. 使用更先进的相似度计算方法，如词嵌入或语义相似度，来改进多样性检查。
3. 实现交互式的创意生成过程，允许用户逐步细化和调整生成的创意。
4. 集成领域特定的知识或规则，以确保生成的创意在特定上下文中是合适和可行的。
5. 实现创意组合技术，将多个 LLM 生成的创意元素智能地组合成更复杂的创意。

通过这些技术，创意生成 Agent 可以利用 LLM 的强大能力来产生高质量、多样化和有针对性的创意。这不仅可以提高创意的数量，还可以显著提升创意的质量和相关性。

## 9.4 创意评估与筛选

创意评估和筛选是确保生成的创意符合质量标准和项目需求的关键步骤。

### 9.4.1 新颖性评估

评估创意的新颖性可以帮助识别真正独特和创新的想法。

示例代码：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

class NoveltyEvaluator:
    def __init__(self, existing_ideas: List[str]):
        self.vectorizer = TfidfVectorizer()
        self.existing_vectors = self.vectorizer.fit_transform(existing_ideas)

    def evaluate_novelty(self, new_idea: str) -> float:
        new_vector = self.vectorizer.transform([new_idea])
        similarities = (self.existing_vectors * new_vector.T).A.flatten()
        return 1 - np.max(similarities)

class CreativityEvaluator:
    def __init__(self, existing_ideas: List[str]):
        self.novelty_evaluator = NoveltyEvaluator(existing_ideas)

    def evaluate_idea(self, idea: str) -> Dict[str, float]:
        novelty_score = self.novelty_evaluator.evaluate_novelty(idea)
        
        # 这里可以添加其他评估指标，如实用性、可行性等
        # 为简化示例，我们只使用新颖性评分
        
        return {
            "novelty": novelty_score,
            "overall_score": novelty_score  # 在实际应用中，这应该是多个指标的加权和
        }

# 使用示例
existing_ideas = [
    "A smartphone that can project holograms",
    "An AI assistant that can read your emotions",
    "A self-cleaning kitchen robot"
]

evaluator = CreativityEvaluator(existing_ideas)

new_ideas = [
    "A mind-reading device for pets",
    "An AI assistant that can read your emotions",
    "A flying car powered by renewable energy"
]

for idea in new_ideas:
    scores = evaluator.evaluate_idea(idea)
    print(f"Idea: {idea}")
    print(f"Novelty score: {scores['novelty']:.2f}")
    print(f"Overall score: {scores['overall_score']:.2f}")
    print()
```

### 9.4.2 实用性分析

评估创意的实用性可以帮助识别那些不仅新颖，而且有实际价值的想法。

示例代码：

```python
import openai
from typing import List, Dict

class UtilityAnalyzer:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def analyze_utility(self, idea: str, context: str) -> Dict[str, float]:
        prompt = f"""
        Idea: {idea}
        Context: {context}

        Analyze the utility of the above idea in the given context. Consider the following aspects:
        1. Feasibility: How easy is it to implement?
        2. Impact: What potential benefits does it offer?
        3. Market demand: Is there a clear need for this idea?

        Provide a score for each aspect (0-10) and an overall utility score (0-10).
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # 解析 LLM 的输出以提取分数
        output = response.choices[0].text.strip()
        scores = self._extract_scores(output)
        return scores

    def _extract_scores(self, text: str) -> Dict[str, float]:
        lines = text.split('\n')
        scores = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':')
                try:
                    scores[key.strip().lower()] = float(value.strip())
                except ValueError:
                    pass
        return scores

# 使用示例
utility_analyzer = UtilityAnalyzer("your_openai_api_key_here")

idea = "A smart mirror that provides personalized health and fitness advice"
context = "Health-conscious consumers looking for convenient ways to monitor their well-being"

utility_scores = utility_analyzer.analyze_utility(idea, context)
print(f"Utility analysis for: {idea}")
for aspect, score in utility_scores.items():
    print(f"{aspect.capitalize()}: {score:.2f}")
```

### 9.4.3 市场潜力预测

预测创意的市场潜力可以帮助识别那些最有可能取得商业成功的想法。

示例代码：

```python
import random
from typing import List, Dict

class MarketPotentialPredictor:
    def __init__(self, market_data: Dict[str, Dict[str, float]]):
        self.market_data = market_data

    def predict_potential(self, idea: str, target_market: str) -> Dict[str, float]:
        # 在实际应用中，这里应该使用更复杂的预测模型
        # 这个简化的版本使用随机值和一些基本规则

        market_size = self.market_data[target_market]["size"]
        growth_rate = self.market_data[target_market]["growth_rate"]
        competition_level = self.market_data[target_market]["competition_level"]

        innovation_factor = random.uniform(0.5, 1.5)
        market_fit = random.uniform(0.3, 1.0)

        potential_score = (market_size * growth_rate * market_fit * innovation_factor) / competition_level
        roi_estimate = potential_score * random.uniform(0.8, 1.2)

        return {
            "market_potential_score": min(potential_score, 10),  # 将分数限制在0-10范围内
            "estimated_roi": roi_estimate,
            "confidence_level": random.uniform(0.6, 0.9)
        }

# 使用示例
market_data = {
    "smart_home": {"size": 7, "growth_rate": 1.2, "competition_level": 0.8},
    "wearable_tech": {"size": 6, "growth_rate": 1.3, "competition_level": 0.7},
    "eco_friendly": {"size": 5, "growth_rate": 1.4, "competition_level": 0.6}
}

predictor = MarketPotentialPredictor(market_data)

ideas = [
    ("A smart mirror that provides personalized health and fitness advice", "smart_home"),
    ("An AI-powered personal stylist wearable", "wearable_tech"),
    ("A self-repairing smartphone screen", "eco_friendly")
]

for idea, market in ideas:
    potential = predictor.predict_potential(idea, market)
    print(f"Market potential for: {idea}")
    print(f"Target market: {market}")
    print(f"Potential score: {potential['market_potential_score']:.2f}")
    print(f"Estimated ROI: {potential['estimated_roi']:.2f}")
    print(f"Confidence level: {potential['confidence_level']:.2f}")
    print()
```

这些评估和筛选组件共同工作，可以帮助创意生成 Agent 识别最有价值的创意：

1. `NoveltyEvaluator` 使用 TF-IDF 和余弦相似度来评估创意的新颖性。
2. `UtilityAnalyzer` 利用 LLM 来分析创意的实用性，考虑可行性、影响力和市场需求。
3. `MarketPotentialPredictor` 使用市场数据和一些启发式规则来预测创意的市场潜力。

在实际应用中，你可能需要：

1. 实现更复杂的新颖性评估方法，如使用词嵌入或深度学习模型来捕捉语义新颖性。
2. 开发更全面的实用性分析框架，可能包括专家评估或用户调研。
3. 使用真实的市场数据和高级预测模型（如时间序列分析或机器学习模型）来改进市场潜力预测。
4. 实现多维度的创意评分系统，综合考虑新颖性、实用性、市场潜力等多个方面。
5. 开发交互式的评估工具，允许人类专家参与评估过程并提供反馈。
6. 实现自动化的创意优化建议，基于评估结果提出改进方向。

通过这些技术，创意生成 Agent 可以不仅产生大量的创意，还能有效地识别和筛选出最有潜力的想法。这种能力对于在创新过程中节省时间和资源，并最大化成功机会至关重要。

## 9.5 人机协作创意优化

人机协作是创意优化过程中的关键环节，它结合了人类的直觉和经验与AI的计算能力和数据处理能力。

### 9.5.1 反馈收集机制

建立有效的反馈收集机制是实现人机协作的基础。

示例代码：

```python
from typing import List, Dict, Any

class FeedbackCollector:
    def __init__(self):
        self.feedback_data = []

    def collect_feedback(self, idea: str, feedback: Dict[str, Any]):
        self.feedback_data.append({
            "idea": idea,
            "feedback": feedback
        })

    def get_feedback_summary(self) -> Dict[str, Any]:
        if not self.feedback_data:
            return {"message": "No feedback collected yet."}

        total_ratings = sum(item['feedback'].get('rating', 0) for item in self.feedback_data)
        avg_rating = total_ratings / len(self.feedback_data)

        common_themes = self._extract_common_themes()

        return {
            "total_feedback": len(self.feedback_data),
            "average_rating": avg_rating,
            "common_themes": common_themes
        }

    def _extract_common_themes(self) -> List[str]:
        # 简化版本：只统计出现频率最高的关键词
        all_comments = ' '.join([item['feedback'].get('comment', '') for item in self.feedback_data])
        words = all_comments.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq, key=word_freq.get, reverse=True)[:5]

# 使用示例
feedback_collector = FeedbackCollector()

# 模拟收集反馈
feedback_collector.collect_feedback(
    "AI-powered personal stylist",
    {"rating": 4, "comment": "Innovative idea, but concerns about privacy"}
)
feedback_collector.collect_feedback(
    "Self-cleaning kitchen robot",
    {"rating": 5, "comment": "Very practical, could save a lot of time"}
)
feedback_collector.collect_feedback(
    "Virtual reality meditation app",
    {"rating": 3, "comment": "Interesting concept, but might be too niche"}
)

# 获取反馈摘要
summary = feedback_collector.get_feedback_summary()
print("Feedback Summary:")
print(f"Total feedback: {summary['total_feedback']}")
print(f"Average rating: {summary['average_rating']:.2f}")
print(f"Common themes: {', '.join(summary['common_themes'])}")
```

### 9.5.2 交互式创意迭代

交互式创意迭代允许人类和AI系统共同改进创意。

示例代码：

```python
import openai
from typing import List, Dict, Any

class InteractiveIdeator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def iterate_idea(self, original_idea: str, human_feedback: str) -> str:
        prompt = f"""
        Original Idea: {original_idea}
        Human Feedback: {human_feedback}

        Based on the original idea and the human feedback, generate an improved version of the idea.
        Ensure that the new idea addresses the feedback while maintaining the core concept.

        Improved Idea:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def brainstorm_variations(self, idea: str, num_variations: int = 3) -> List[str]:
        prompt = f"""
        Original Idea: {idea}

        Generate {num_variations} variations of this idea, each exploring a different aspect or approach:

        Variations:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.8,
        )

        variations = response.choices[0].text.strip().split('\n')
        return [v.strip() for v in variations if v.strip()]

# 使用示例
ideator = InteractiveIdeator("your_openai_api_key_here")

original_idea = "A smart mirror that provides personalized health and fitness advice"
human_feedback = "Great idea, but concerns about privacy and data security"

improved_idea = ideator.iterate_idea(original_idea, human_feedback)
print(f"Improved Idea: {improved_idea}")

variations = ideator.brainstorm_variations(improved_idea)
print("\nIdea Variations:")
for i, variation in enumerate(variations, 1):
    print(f"{i}. {variation}")
```

### 9.5.3 创意组合与融合

创意组合和融合可以产生更加新颖和综合的想法。

示例代码：

```python
import random
from typing import List, Dict, Any

class CreativeCombiner:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def combine_ideas(self, idea1: str, idea2: str) -> str:
        prompt = f"""
        Idea 1: {idea1}
        Idea 2: {idea2}

        Create a new, innovative idea by combining elements from both ideas above.
        The new idea should leverage the strengths of both original ideas and create something unique.

        Combined Idea:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.8,
        )

        return response.choices[0].text.strip()

    def fusion_brainstorm(self, ideas: List[str], num_fusions: int = 3) -> List[str]:
        fusions = []
        for _ in range(num_fusions):
            idea1, idea2 = random.sample(ideas, 2)
            fusion = self.combine_ideas(idea1, idea2)
            fusions.append(fusion)
        return fusions

# 使用示例
combiner = CreativeCombiner("your_openai_api_key_here")

ideas = [
    "A smart mirror that provides personalized health advice",
    "An AI-powered personal stylist wearable",
    "A self-repairing smartphone screen",
    "Virtual reality meditation app"
]

# 组合两个特定的创意
combined_idea = combiner.combine_ideas(ideas[0], ideas[1])
print(f"Combined Idea: {combined_idea}")

# 生成多个融合创意
fusion_ideas = combiner.fusion_brainstorm(ideas)
print("\nFusion Ideas:")
for i, fusion in enumerate(fusion_ideas, 1):
    print(f"{i}. {fusion}")
```

这些人机协作创意优化组件共同工作，可以显著提升创意的质量和相关性：

1. `FeedbackCollector` 收集和汇总人类对创意的反馈。
2. `InteractiveIdeator` 允许基于人类反馈迭代改进创意，并生成创意变体。
3. `CreativeCombiner` 通过组合不同创意的元素来产生新的综合创意。

在实际应用中，你可能需要：

1. 实现更复杂的反馈分析算法，如情感分析或主题建模，以更好地理解人类反馈。
2. 开发一个用户友好的界面，方便人类专家和AI系统进行实时交互和创意迭代。
3. 使用更高级的自然语言生成技术，如GPT-3或其他大型语言模型，来提高创意组合和融合的质量。
4. 实现创意版本控制系统，跟踪创意的演变过程，并允许回溯到之前的版本。
5. 开发创意评估指标，以量化人机协作过程对创意质量的提升。
6. 实现个性化的创意优化建议，基于特定用户或团队的偏好和历史反馈。

通过这些技术，创意生成Agent可以与人类专家紧密协作，充分利用双方的优势。人类可以提供直觉、领域知识和创造性思维，而AI系统可以处理大量数据、生成多样化的创意，并快速迭代。这种协作可以产生比单独使用任何一方都更优秀的创意成果。

## 9.6 创意展示与应用

创意的有效展示和实际应用是将创新想法转化为现实的关键步骤。

### 9.6.1 多模态创意呈现

多模态呈现可以更全面、生动地展示创意，增强受众的理解和参与度。

示例代码：

```python
import openai
from PIL import Image
import requests
from io import BytesIO
import base64

class MultiModalPresenter:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_text_description(self, idea: str) -> str:
        prompt = f"""
        Create a compelling and detailed description for the following idea:
        {idea}

        The description should highlight key features, potential benefits, and target audience.
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def generate_image(self, idea: str) -> Image.Image:
        response = openai.Image.create(
            prompt=f"A visual representation of: {idea}",
            n=1,
            size="512x512"
        )

        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))

    def create_presentation(self, idea: str) -> Dict[str, Any]:
        description = self.generate_text_description(idea)
        image = self.generate_image(idea)

        # 将图像转换为base64字符串
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "idea": idea,
            "description": description,
            "image": img_str
        }

# 使用示例
presenter = MultiModalPresenter("your_openai_api_key_here")

idea = "A smart garden that automatically adjusts watering and nutrients based on plant health"
presentation = presenter.create_presentation(idea)

print(f"Idea: {presentation['idea']}")
print(f"\nDescription: {presentation['description']}")
print("\nImage has been generated and encoded as base64.")
```

### 9.6.2 创意原型快速生成

快速原型可以帮助验证创意的可行性，并为进一步开发提供基础。

示例代码：

```python
from typing import List, Dict, Any

class RapidPrototyper:
    def generate_mockup(self, idea: str) -> str:
        # 在实际应用中，这里可能会调用设计工具的API来生成UI mockup
        # 这里我们用文本描述来模拟mockup生成
        return f"[Mockup for {idea}: A simple user interface with main features outlined]"

    def create_feature_list(self, idea: str) -> List[str]:
        # 在实际应用中，这可能会使用NLP来分析创意描述并提取特性
        # 这里我们使用一个简化的方法
        return [
            "User authentication and profile management",
            "Main functionality implementation",
            "Data storage and retrieval",
            "User interface for interaction",
            "Integration with external services or APIs"
        ]

    def estimate_development_time(self, features: List[str]) -> Dict[str, int]:
        # 这是一个非常简化的时间估算方法
        return {
            "design": len(features) * 2,  # 每个特性2天设计时间
            "development": len(features) * 5,  # 每个特性5天开发时间
            "testing": len(features) * 3  # 每个特性3天测试时间
        }

    def generate_prototype_plan(self, idea: str) -> Dict[str, Any]:
        mockup = self.generate_mockup(idea)
        features = self.create_feature_list(idea)
        time_estimate = self.estimate_development_time(features)

        return {
            "idea": idea,
            "mockup": mockup,
            "features": features,
            "time_estimate": time_estimate,
            "total_time": sum(time_estimate.values())
        }

# 使用示例
prototyper = RapidPrototyper()

idea = "A smart garden that automatically adjusts watering and nutrients based on plant health"
prototype_plan = prototyper.generate_prototype_plan(idea)

print(f"Prototype Plan for: {prototype_plan['idea']}")
print(f"\nMockup: {prototype_plan['mockup']}")
print("\nFeatures:")
for feature in prototype_plan['features']:
    print(f"- {feature}")
print("\nTime Estimate:")
for phase, days in prototype_plan['time_estimate'].items():
    print(f"- {phase.capitalize()}: {days} days")
print(f"\nTotal Estimated Time: {prototype_plan['total_time']} days")
```

### 9.6.3 版权保护与管理

保护创意的知识产权对于维护创新者的权益至关重要。

示例代码：

```python
import hashlib
import time
from typing import Dict, Any

class CopyrightManager:
    def __init__(self):
        self.copyright_registry = {}

    def register_copyright(self, idea: str, author: str) -> Dict[str, Any]:
        timestamp = time.time()
        idea_hash = hashlib.sha256(f"{idea}{author}{timestamp}".encode()).hexdigest()

        copyright_info = {
            "idea": idea,
            "author": author,
            "timestamp": timestamp,
            "copyright_id": idea_hash[:10]
        }

        self.copyright_registry[idea_hash] = copyright_info
        return copyright_info

    def verify_copyright(self, copyright_id: str) -> Dict[str, Any]:
        for idea_hash, info in self.copyright_registry.items():
            if info['copyright_id'] == copyright_id:
                return info
        return {"error": "Copyright not found"}

    def generate_certificate(self, copyright_info: Dict[str, Any]) -> str:
        return f"""
        Copyright Certificate

        Idea: {copyright_info['idea']}
        Author: {copyright_info['author']}
        Registration Date: {time.ctime(copyright_info['timestamp'])}
        Copyright ID: {copyright_info['copyright_id']}

        This certifies that the above idea has been registered in our copyright system.
        """

# 使用示例copyright_manager = CopyrightManager()

idea = "A smart garden that automatically adjusts watering and nutrients based on plant health"
author = "Jane Doe"

# 注册版权
copyright_info = copyright_manager.register_copyright(idea, author)
print("Copyright Registered:")
print(f"Copyright ID: {copyright_info['copyright_id']}")

# 验证版权
verification = copyright_manager.verify_copyright(copyright_info['copyright_id'])
if 'error' not in verification:
    print("\nCopyright Verified:")
    print(f"Idea: {verification['idea']}")
    print(f"Author: {verification['author']}")
    print(f"Registration Date: {time.ctime(verification['timestamp'])}")

# 生成证书
certificate = copyright_manager.generate_certificate(copyright_info)
print("\nCopyright Certificate:")
print(certificate)
```

这些创意展示与应用组件共同工作，可以帮助将创意转化为可视化、可验证和可保护的资产：

1. `MultiModalPresenter` 生成创意的文本描述和视觉表现，提供全面的创意展示。
2. `RapidPrototyper` 快速生成创意的原型计划，包括功能列表和时间估算。
3. `CopyrightManager` 管理创意的版权，提供注册、验证和证书生成功能。

在实际应用中，你可能需要：

1. 集成更高级的图像生成AI，如DALL-E 2或Midjourney，以创建更精细和定制化的创意视觉表现。
2. 开发交互式的原型工具，允许用户在线上快速构建和测试创意的基本功能。
3. 实现更复杂的版权管理系统，可能包括区块链技术来确保版权信息的不可篡改性。
4. 创建一个创意展示平台，允许创作者展示他们的创意，并与潜在的投资者或合作伙伴互动。
5. 开发创意评估工具，帮助用户评估他们的创意在市场中的潜力和竞争力。
6. 实现自动化的专利检索功能，帮助创作者确保他们的创意不会侵犯现有的专利。
7. 创建创意众筹或众包平台，允许创作者寻求资金支持或技术合作来实现他们的创意。

通过这些技术，创意生成Agent不仅可以产生创新想法，还能帮助将这些想法转化为可视化的概念、可行的原型计划，并提供必要的知识产权保护。这种全面的支持可以大大提高创新过程的效率，并增加创意成功商业化的机会。
