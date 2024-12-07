
# 第8章：任务自动化 Agent 开发实践

任务自动化 Agent 是一种能够自主执行复杂任务序列的 AI 系统。它能理解高级指令，分解任务，并利用各种工具和 API 来完成目标。本章将探讨如何开发这样一个系统。

## 8.1 系统需求与架构设计

### 8.1.1 自动化需求分析

首先，我们需要分析自动化的需求，确定系统应该能够处理的任务类型和范围。

示例代码（需求收集和分析工具）：

```python
from typing import List, Dict
from collections import Counter

class AutomationRequirementAnalyzer:
    def __init__(self):
        self.requirements = []

    def add_requirement(self, task_type: str, description: str, priority: int):
        self.requirements.append({
            "task_type": task_type,
            "description": description,
            "priority": priority
        })

    def analyze_requirements(self) -> Dict[str, Any]:
        task_types = Counter([req["task_type"] for req in self.requirements])
        priorities = Counter([req["priority"] for req in self.requirements])
        
        return {
            "total_requirements": len(self.requirements),
            "task_type_distribution": dict(task_types),
            "priority_distribution": dict(priorities),
            "high_priority_tasks": [req for req in self.requirements if req["priority"] == 1]
        }

# 使用示例
analyzer = AutomationRequirementAnalyzer()

analyzer.add_requirement("data_processing", "Extract data from CSV files", 2)
analyzer.add_requirement("scheduling", "Schedule meetings based on availability", 1)
analyzer.add_requirement("reporting", "Generate monthly sales reports", 2)
analyzer.add_requirement("communication", "Send automated email notifications", 1)

analysis_result = analyzer.analyze_requirements()
print(f"Total requirements: {analysis_result['total_requirements']}")
print(f"Task type distribution: {analysis_result['task_type_distribution']}")
print(f"Priority distribution: {analysis_result['priority_distribution']}")
print("High priority tasks:")
for task in analysis_result['high_priority_tasks']:
    print(f"- {task['description']}")
```

### 8.1.2 任务类型与流程梳理

接下来，我们需要梳理出系统需要支持的任务类型和它们的典型流程。

示例代码（任务流程定义工具）：

```python
from typing import List, Dict

class TaskFlow:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.steps = []

    def add_step(self, step_name: str, description: str):
        self.steps.append({"name": step_name, "description": description})

    def get_flow(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "steps": self.steps
        }

class TaskFlowManager:
    def __init__(self):
        self.task_flows = {}

    def define_task_flow(self, task_flow: TaskFlow):
        self.task_flows[task_flow.task_type] = task_flow

    def get_task_flow(self, task_type: str) -> TaskFlow:
        return self.task_flows.get(task_type)

    def list_task_types(self) -> List[str]:
        return list(self.task_flows.keys())

# 使用示例
flow_manager = TaskFlowManager()

# 定义数据处理任务流程
data_processing_flow = TaskFlow("data_processing")
data_processing_flow.add_step("data_extraction", "Extract data from source files")
data_processing_flow.add_step("data_cleaning", "Clean and preprocess the extracted data")
data_processing_flow.add_step("data_analysis", "Perform analysis on the cleaned data")
data_processing_flow.add_step("report_generation", "Generate a report based on the analysis")

flow_manager.define_task_flow(data_processing_flow)

# 定义调度任务流程
scheduling_flow = TaskFlow("scheduling")
scheduling_flow.add_step("availability_check", "Check availability of all participants")
scheduling_flow.add_step("time_slot_selection", "Select the best time slot")
scheduling_flow.add_step("invitation_sending", "Send meeting invitations")
scheduling_flow.add_step("confirmation_collection", "Collect confirmations from participants")

flow_manager.define_task_flow(scheduling_flow)

# 使用定义的任务流程
print("Supported task types:", flow_manager.list_task_types())

data_processing_steps = flow_manager.get_task_flow("data_processing").get_flow()
print("\nData Processing Flow:")
for step in data_processing_steps["steps"]:
    print(f"- {step['name']}: {step['description']}")
```

### 8.1.3 系统模块设计

基于需求分析和任务流程，我们可以设计系统的主要模块。

示例代码（系统模块设计框架）：

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class Module(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class TaskParser(Module):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 解析用户输入，识别任务类型和参数
        return {"task_type": "data_processing", "parameters": {"file": "data.csv"}}

class TaskPlanner(Module):
    def __init__(self, flow_manager: TaskFlowManager):
        self.flow_manager = flow_manager

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task_type = input_data["task_type"]
        task_flow = self.flow_manager.get_task_flow(task_type)
        return {"plan": task_flow.get_flow()}

class TaskExecutor(Module):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 执行任务计划中的每个步骤
        return {"status": "completed", "result": "Task executed successfully"}

class ResultFormatter(Module):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 格式化任务执行结果
        return {"formatted_result": "Formatted output of the task execution"}

class AutomationSystem:
    def __init__(self, flow_manager: TaskFlowManager):
        self.task_parser = TaskParser()
        self.task_planner = TaskPlanner(flow_manager)
        self.task_executor = TaskExecutor()
        self.result_formatter = ResultFormatter()

    def run(self, user_input: str) -> str:
        parsed_input = self.task_parser.process({"raw_input": user_input})
        task_plan = self.task_planner.process(parsed_input)
        execution_result = self.task_executor.process(task_plan)
        formatted_result = self.result_formatter.process(execution_result)
        return formatted_result["formatted_result"]

# 使用示例
flow_manager = TaskFlowManager()
# 假设我们已经定义了任务流程，如前面的例子所示

automation_system = AutomationSystem(flow_manager)
result = automation_system.run("Process the sales data from last month")
print(result)
```

这个系统架构设计提供了一个模块化的框架，可以轻松扩展和修改各个组件。在实际应用中，你可能需要进一步细化每个模块的功能，并添加更多的模块来处理诸如错误处理、日志记录、安全认证等方面。

## 8.2 任务理解与规划

### 8.2.1 自然语言指令解析

自然语言指令解析是将用户的自然语言输入转换为系统可以理解和执行的结构化任务描述的过程。

示例代码（使用简单的规则基础方法）：

```python
import re
from typing import Dict, Any

class NLParser:
    def __init__(self):
        self.task_patterns = {
            "data_processing": r"process|analyze|extract data from (\w+)",
            "scheduling": r"schedule|arrange|set up (a meeting|an appointment)",
            "reporting": r"generate|create|produce (a|an) (\w+) report",
        }

    def parse(self, input_text: str) -> Dict[str, Any]:
        for task_type, pattern in self.task_patterns.items():
            match = re.search(pattern, input_text, re.IGNORECASE)
            if match:
                return {
                    "task_type": task_type,
                    "parameters": self._extract_parameters(task_type, match)
                }
        return {"task_type": "unknown", "parameters": {}}

    def _extract_parameters(self, task_type: str, match: re.Match) -> Dict[str, str]:
        if task_type == "data_processing":
            return {"source": match.group(1)}
        elif task_type == "scheduling":
            return {"event_type": match.group(1)}
        elif task_type == "reporting":
            return {"report_type": match.group(2)}
        return {}

# 使用示例
parser = NLParser()

inputs = [
    "Can you process the data from sales.csv?",
    "I need to schedule a meeting with the marketing team",
    "Please generate a monthly financial report",
]

for input_text in inputs:
    result = parser.parse(input_text)
    print(f"Input: {input_text}")
    print(f"Parsed result: {result}")
    print("---")
```

### 8.2.2 任务可行性分析

任务可行性分析涉及评估系统是否有能力执行给定的任务，以及执行任务所需的资源。

示例代码：

```python
from typing import Dict, Any, List

class TaskAnalyzer:
    def __init__(self):
        self.available_resources = {
            "data_processing": ["CSV", "JSON", "XML"],
            "scheduling": ["internal", "external"],
            "reporting": ["financial", "sales", "marketing"]
        }
        self.required_permissions = {
            "data_processing": ["read_files", "write_files"],
            "scheduling": ["calendar_access", "email_access"],
            "reporting": ["database_access", "file_creation"]
        }

    def analyze_feasibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = task["task_type"]
        parameters = task["parameters"]

        if task_type not in self.available_resources:
            return {"feasible": False, "reason": "Unsupported task type"}

        resource_check = self._check_resources(task_type, parameters)
        permission_check = self._check_permissions(task_type)

        if not resource_check["available"]:
            return {"feasible": False, "reason": resource_check["reason"]}
        
        if not permission_check["granted"]:
            return {"feasible": False, "reason": permission_check["reason"]}

        return {
            "feasible": True,
            "required_resources": resource_check["required"],
            "required_permissions": permission_check["required"]
        }

    def _check_resources(self, task_type: str, parameters: Dict[str, str]) -> Dict[str, Any]:
        if task_type == "data_processing":
            source_type = parameters.get("source", "").split(".")[-1].upper()
            if source_type not in self.available_resources[task_type]:
                return {"available": False, "reason": f"Unsupported data source type: {source_type}"}
            return {"available": True, "required": [source_type]}
        
        # Add similar checks for other task types
        return {"available": True, "required": self.available_resources[task_type]}

    def _check_permissions(self, task_type: str) -> Dict[str, Any]:
        # In a real system, this would check against user permissions
        return {"granted": True, "required": self.required_permissions[task_type]}

# 使用示例
analyzer = TaskAnalyzer()

tasks = [
    {"task_type": "data_processing", "parameters": {"source": "sales.csv"}},
    {"task_type": "scheduling", "parameters": {"event_type": "meeting"}},
    {"task_type": "reporting", "parameters": {"report_type": "financial"}},
    {"task_type": "data_processing", "parameters": {"source": "data.pdf"}},
]

for task in tasks:
    result = analyzer.analyze_feasibility(task)
    print(f"Task: {task}")
    print(f"Feasibility analysis: {result}")
    print("---")
```

### 8.2.3 子任务生成与排序

子任务生成与排序涉及将复杂任务分解为可管理的步骤，并确定这些步骤的最佳执行顺序。

示例代码：

```python
from typing import List, Dict, Any
import networkx as nx

class SubtaskGenerator:
    def __init__(self, flow_manager: TaskFlowManager):
        self.flow_manager = flow_manager

    def generate_subtasks(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_type = task["task_type"]
        task_flow = self.flow_manager.get_task_flow(task_type)
        
        if not task_flow:
            return []

        subtasks = []
        for step in task_flow.steps:
            subtask = {
                "name": step["name"],
                "description": step["description"],
                "parameters": task["parameters"]
            }
            subtasks.append(subtask)

        return subtasks

class SubtaskSorter:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()

    def add_dependency(self, task1: str, task2: str):
        self.dependency_graph.add_edge(task1, task2)

    def sort_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        task_names = [subtask["name"] for subtask in subtasks]
        try:
            sorted_names = list(nx.topological_sort(self.dependency_graph.subgraph(task_names)))
            return sorted(subtasks, key=lambda x: sorted_names.index(x["name"]))
        except nx.NetworkXUnfeasible:
            # If there's a cycle in the dependencies, return the original order
            return subtasks

# 使用示例
flow_manager = TaskFlowManager()
# 假设我们已经定义了任务流程，如前面的例子所示

subtask_generator = SubtaskGenerator(flow_manager)
subtask_sorter = SubtaskSorter()

# 添加一些依赖关系
subtask_sorter.add_dependency("data_extraction", "data_cleaning")
subtask_sorter.add_dependency("data_cleaning", "data_analysis")
subtask_sorter.add_dependency("data_analysis", "report_generation")

task = {"task_type": "data_processing", "parameters": {"source": "sales.csv"}}
subtasks = subtask_generator.generate_subtasks(task)
sorted_subtasks = subtask_sorter.sort_subtasks(subtasks)

print("Generated and sorted subtasks:")
for subtask in sorted_subtasks:
    print(f"- {subtask['name']}: {subtask['description']}")
```

这些组件共同工作，可以帮助系统理解用户的自然语言指令，评估任务的可行性，并生成一个结构化的、有序的子任务列表。在实际应用中，你可能需要使用更复杂的自然语言处理技术，如命名实体识别或依存句法分析，来提高指令解析的准确性。同时，任务可行性分析可能需要考虑更多因素，如系统负载、用户权限等。子任务生成和排序可能需要更复杂的规则或机器学习模型来处理各种边缘情况和复杂的任务依赖关系。

## 8.3 执行环境集成

执行环境集成是任务自动化 Agent 的关键组成部分，它使 Agent 能够与各种系统和服务交互，执行实际的任务。

### 8.3.1 操作系统接口

操作系统接口允许 Agent 执行文件操作、进程管理等系统级任务。

示例代码：

```python
import os
import subprocess
from typing import List, Dict, Any

class OSInterface:
    def list_directory(self, path: str) -> List[str]:
        return os.listdir(path)

    def create_directory(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except OSError:
            return False

    def delete_file(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    def execute_command(self, command: List[str]) -> Dict[str, Any]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return {"success": True, "output": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": str(e), "output": e.stdout}

# 使用示例
os_interface = OSInterface()

# 列出目录内容
print("Directory contents:", os_interface.list_directory("."))

# 创建新目录
if os_interface.create_directory("./new_folder"):
    print("New directory created successfully")

# 执行系统命令
result = os_interface.execute_command(["echo", "Hello, World!"])
if result["success"]:
    print("Command output:", result["output"])
else:
    print("Command failed:", result["error"])
```

### 8.3.2 应用程序 API 集成

应用程序 API 集成使 Agent 能够与各种外部服务和应用程序交互。

示例代码（以天气 API 为例）：

```python
import requests
from typing import Dict, Any

class WeatherAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city: str) -> Dict[str, Any]:
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"]
            }
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

class EmailAPI:
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        # This is a simplified example. In a real application, you would use
        # Python's smtplib to actually send the email.
        print(f"Sending email to {to} with subject '{subject}'")
        print(f"Email body: {body}")
        return {"success": True, "message": "Email sent successfully"}

# 使用示例
weather_api = WeatherAPI("your_api_key_here")
email_api = EmailAPI("smtp.example.com", 587, "username", "password")

# 获取天气信息
weather = weather_api.get_weather("London")
if weather["success"]:
    print(f"Current weather in London: {weather['temperature']}°C, {weather['description']}")

# 发送邮件
email_result = email_api.send_email(
    "recipient@example.com",
    "Weather Report",
    f"The current weather in London is {weather['temperature']}°C, {weather['description']}"
)
if email_result["success"]:
    print("Email sent successfully")
```

### 8.3.3 网络爬虫与数据采集

网络爬虫使 Agent 能够从网页中提取信息，这对于数据收集和分析任务非常有用。

示例代码：

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def scrape_webpage(self, url: str) -> Dict[str, Any]:
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return {
                "success": True,
                "title": soup.title.string if soup.title else "",
                "text": soup.get_text(),
                "links": [a['href'] for a in soup.find_all('a', href=True)]
            }
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def extract_data(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            data = {}
            for key, selector in selectors.items():
                elements = soup.select(selector)
                data[key] = [elem.get_text(strip=True) for elem in elements]
            return {"success": True, "data": data}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

# 使用示例
scraper = WebScraper()

# 抓取网页内容
result = scraper.scrape_webpage("https://example.com")
if result["success"]:
    print(f"Page title: {result['title']}")
    print(f"Number of links found: {len(result['links'])}")

# 提取特定数据
selectors = {
    "headlines": "h2.headline",
    "article_dates": "span.article-date"
}
data_result = scraper.extract_data("https://example-news-site.com", selectors)
if data_result["success"]:
    for headline, date in zip(data_result["data"]["headlines"], data_result["data"]["article_dates"]):
        print(f"Headline: {headline}, Date: {date}")
```

通过这些接口，任务自动化 Agent 可以与操作系统、外部 API 和网页进行交互，执行各种复杂的任务。在实际应用中，你可能需要添加更多的错误处理、重试机制和安全措施。此外，对于网络爬虫，请确保遵守网站的使用条款和 robots.txt 文件的规定。

## 8.4 LLM 辅助决策

大型语言模型（LLM）可以帮助任务自动化 Agent 处理复杂的决策场景，特别是在处理不确定性和异常情况时。

### 8.4.1 不确定性处理

当 Agent 遇到不明确的指令或多种可能的执行路径时，LLM 可以帮助解释和选择最佳行动。

示例代码：

```python
import openai
from typing import List, Dict, Any

class LLMDecisionMaker:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def resolve_ambiguity(self, context: str, options: List[str]) -> str:
        prompt = f"""
        Context: {context}
        
        Options:
        {'\n'.join([f'{i+1}. {option}' for i, option in enumerate(options)])}
        
        Based on the context, which option is the most appropriate? Explain your reasoning.
        """
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        return response.choices[0].text.strip()

# 使用示例
llm_decider = LLMDecisionMaker("your_openai_api_key_here")

context = "The user has requested to 'send the file', but multiple files are available."
options = [
    "Send the most recently modified file",
    "Ask the user to specify which file",
    "Send all available files"
]

decision = llm_decider.resolve_ambiguity(context, options)
print("LLM Decision:")
print(decision)
```

### 8.4.2 异常情况应对

当任务执行过程中遇到异常或错误时，LLM 可以帮助生成适当的响应或提出解决方案。

示例代码：

```python
class LLMErrorHandler:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def handle_error(self, error_message: str, task_context: str) -> str:
        prompt = f"""
        Error: {error_message}
        Task Context: {task_context}
        
        Given the error and the context of the task, suggest a solution or next steps to resolve the issue. 
        Provide a clear and concise response that an automated system could potentially act upon.
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

# 使用示例
llm_error_handler = LLMErrorHandler("your_openai_api_key_here")

error_message = "FileNotFoundError: The specified file 'report.pdf' does not exist."
task_context = "The system was attempting to email the monthly report to stakeholders."

solution = llm_error_handler.handle_error(error_message, task_context)
print("LLM Suggested Solution:")
print(solution)
```

### 8.4.3 结果验证与纠错

LLM 可以帮助验证任务执行的结果，并在必要时提出纠正建议。

示例代码：

```python
class LLMResultValidator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def validate_result(self, task_description: str, expected_outcome: str, actual_result: str) -> Dict[str, Any]:
        prompt = f"""
        Task Description: {task_description}
        Expected Outcome: {expected_outcome}
        Actual Result: {actual_result}
        
        Analyze the actual result against the expected outcome for the given task. 
        Determine if the result is correct, and if not, suggest corrections. Provide your response in the following format:
        
        Is Correct: [Yes/No]
        Explanation: [Your explanation]
        Suggested Correction: [Your suggestion if applicable, or "N/A" if the result is correct]
        """
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        validation_result = response.choices[0].text.strip()
        
        # Parse the validation result
        lines = validation_result.split('\n')
        is_correct = lines[0].split(':')[1].strip().lower() == 'yes'
        explanation = lines[1].split(':')[1].strip()
        suggested_correction = lines[2].split(':')[1].strip()
        
        return {
            "is_correct": is_correct,
            "explanation": explanation,
            "suggested_correction": suggested_correction if suggested_correction != "N/A" else None
        }

# 使用示例
llm_validator = LLMResultValidator("your_openai_api_key_here")

task_description = "Calculate the average of the numbers: 10, 15, 20, 25, 30"
expected_outcome = "The average should be 20"
actual_result = "The calculated average is 21"

validation = llm_validator.validate_result(task_description, expected_outcome, actual_result)
print("Validation Result:")
print(f"Is Correct: {validation['is_correct']}")
print(f"Explanation: {validation['explanation']}")
if validation['suggested_correction']:
    print(f"Suggested Correction: {validation['suggested_correction']}")
```

这些 LLM 辅助决策组件可以显著提高任务自动化 Agent 的智能性和适应性。它们可以帮助 Agent 处理模糊的指令、应对异常情况，并确保任务结果的准确性。在实际应用中，你可能需要进一步优化提示工程，以获得更精确和可靠的 LLM 输出。同时，考虑到 API 调用的成本和延迟，你可能需要实现缓存机制或设置使用阈值，以平衡性能和成本。

## 8.5 执行监控与报告

执行监控和报告是任务自动化 Agent 的重要组成部分，它们确保任务的顺利进行，并为用户提供清晰的执行状态和结果。

### 8.5.1 实时状态跟踪

实时状态跟踪允许 Agent 和用户了解任务的进展情况。

示例代码：

```python
import time
from typing import Dict, Any, List
from enum import Enum

class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskTracker:
    def __init__(self):
        self.tasks = {}

    def create_task(self, task_id: str, description: str) -> None:
        self.tasks[task_id] = {
            "description": description,
            "status": TaskStatus.PENDING,
            "progress": 0,
            "start_time": None,
            "end_time": None,
            "subtasks": []
        }

    def update_task_status(self, task_id: str, status: TaskStatus, progress: int = None) -> None:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        self.tasks[task_id]["status"] = status
        if progress is not None:
            self.tasks[task_id]["progress"] = progress

        if status == TaskStatus.IN_PROGRESS and self.tasks[task_id]["start_time"] is None:
            self.tasks[task_id]["start_time"] = time.time()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.tasks[task_id]["end_time"] = time.time()

    def add_subtask(self, task_id: str, subtask: str) -> None:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        self.tasks[task_id]["subtasks"].append({"description": subtask, "completed": False})

    def complete_subtask(self, task_id: str, subtask_index: int) -> None:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        if subtask_index >= len(self.tasks[task_id]["subtasks"]):
            raise ValueError(f"Subtask index {subtask_index} out of range")
        self.tasks[task_id]["subtasks"][subtask_index]["completed"] = True

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.tasks[task_id]

# 使用示例
tracker = TaskTracker()

# 创建任务
tracker.create_task("task1", "Process monthly sales data")

# 更新任务状态
tracker.update_task_status("task1", TaskStatus.IN_PROGRESS)

# 添加子任务
tracker.add_subtask("task1", "Extract data from database")
tracker.add_subtask("task1", "Clean and preprocess data")
tracker.add_subtask("task1", "Generate sales report")

# 完成子任务
tracker.complete_subtask("task1", 0)
tracker.update_task_status("task1", TaskStatus.IN_PROGRESS, 33)

# 获取任务状态
status = tracker.get_task_status("task1")
print(f"Task: {status['description']}")
print(f"Status: {status['status'].value}")
print(f"Progress: {status['progress']}%")
print("Subtasks:")
for i, subtask in enumerate(status['subtasks']):
    print(f"  {i+1}. {subtask['description']} - {'Completed' if subtask['completed'] else 'Pending'}")
```

### 8.5.2 执行日志与数据收集

执行日志和数据收集对于调试、优化和审计非常重要。

示例代码：

```python
import logging
from typing import Dict, Any
import json
from datetime import datetime

class ExecutionLogger:
    def __init__(self, log_file: str):
        self.logger = logging.getLogger("ExecutionLogger")
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.logger.info(json.dumps(log_entry))

    def log_error(self, error_message: str, details: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "ERROR",
            "error_message": error_message,
            "details": details
        }
        self.logger.error(json.dumps(log_entry))

class DataCollector:
    def __init__(self, storage_file: str):
        self.storage_file = storage_file

    def store_data(self, data: Dict[str, Any]) -> None:
        with open(self.storage_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')

    def retrieve_data(self, filter_func=None):
        with open(self.storage_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if filter_func is None or filter_func(data):
                    yield data

# 使用示例
logger = ExecutionLogger("execution_log.txt")
data_collector = DataCollector("collected_data.jsonl")

# 记录事件
logger.log_event("TASK_STARTED", {"task_id": "task1", "description": "Process monthly sales data"})

# 记录错误
try:
    # 模拟一个操作
    raise ValueError("Invalid data format")
except Exception as e:
    logger.log_error(str(e), {"task_id": "task1", "step": "data_preprocessing"})

# 收集数据
data_collector.store_data({
    "task_id": "task1",
    "timestamp": datetime.now().isoformat(),
    "sales_total": 15000,
    "top_product": "Widget A"
})

# 检索数据
for entry in data_collector.retrieve_data(lambda x: x["task_id"] == "task1"):
    print(entry)
```

### 8.5.3 结果分析与报告生成

结果分析和报告生成帮助用户理解任务执行的结果和影响。

示例代码：

```python
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class ResultAnalyzer:
    def analyze_sales_data(self, sales_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_sales = sum(item['amount'] for item in sales_data)
        average_sale = total_sales / len(sales_data)
        top_products = sorted(sales_data, key=lambda x: x['amount'], reverse=True)[:5]
        
        return {
            "total_sales": total_sales,
            "average_sale": average_sale,
            "top_products": top_products
        }

class ReportGenerator:
    def __init__(self):
        self.result_analyzer = ResultAnalyzer()

    def generate_sales_report(self, sales_data: List[Dict[str, Any]]) -> str:
        analysis = self.result_analyzer.analyze_sales_data(sales_data)
        
        report = f"""
        Sales Report
        ============

        Total Sales: ${analysis['total_sales']:.2f}
        Average Sale: ${analysis['average_sale']:.2f}

        Top 5 Products:
        """
        
        for product in analysis['top_products']:
            report += f"- {product['name']}: ${product['amount']:.2f}\n"
        
        # Generate a bar chart of top products
        plt.figure(figsize=(10, 5))
        plt.bar([p['name'] for p in analysis['top_products']], [p['amount'] for p in analysis['top_products']])
        plt.title("Top 5 Products by Sales")
        plt.xlabel("Product")
        plt.ylabel("Sales Amount ($)")
        plt.xticks(rotation=45, ha='right')
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        report += f"\n![Top 5 Products](data:image/png;base64,{image_base64})"
        
        return report

# 使用示例
sales_data = [
    {"name": "Product A", "amount": 1000},
    {"name": "Product B", "amount": 1500},
    {"name": "Product C", "amount": 800},
    {"name": "Product D", "amount": 1200},
    {"name": "Product E", "amount": 950},
]

report_generator = ReportGenerator()
report = report_generator.generate_sales_report(sales_data)
print(report)
```

这些组件共同工作，可以提供全面的任务执行监控和报告功能：

1. `TaskTracker` 允许实时跟踪任务和子任务的进度。
2. `ExecutionLogger` 和 `DataCollector` 记录详细的执行日志和收集相关数据。
3. `ResultAnalyzer` 和 `ReportGenerator` 分析执行结果并生成易于理解的报告。

在实际应用中，你可能需要：

1. 实现更复杂的数据分析算法，以提供更深入的洞察。
2. 添加可视化组件，如交互式图表或仪表板，以更直观地展示任务进度和结果。
3. 实现实时通知机制，在关键事件发生时（如任务完成或错误发生）立即通知相关人员。
4. 集成机器学习模型，以预测任务执行时间或识别潜在的问题。
5. 实现自定义报告模板，以满足不同用户或部门的需求。

通过这些功能，任务自动化 Agent 可以提供透明、可追踪的执行过程，并生成有价值的分析报告，帮助用户更好地理解和优化自动化流程。

## 8.6 安全性与权限管理

在开发任务自动化 Agent 时，确保系统的安全性和适当的权限管理至关重要，特别是当 Agent 可能访问敏感数据或执行关键操作时。

### 8.6.1 身份认证与授权

实现强大的身份认证和授权机制是保护系统安全的第一道防线。

示例代码：

```python
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, user_id: str, role: str) -> str:
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

class PermissionManager:
    def __init__(self):
        self.role_permissions = {
            'admin': ['read', 'write', 'execute'],
            'user': ['read', 'execute'],
            'guest': ['read']
        }

    def check_permission(self, role: str, required_permission: str) -> bool:
        if role not in self.role_permissions:
            return False
        return required_permission in self.role_permissions[role]

# 使用示例
auth_manager = AuthManager("your-secret-key")
permission_manager = PermissionManager()

# 生成令牌
user_token = auth_manager.generate_token("user123", "user")

# 验证令牌并检查权限
try:
    payload = auth_manager.verify_token(user_token)
    if permission_manager.check_permission(payload['role'], 'read'):
        print(f"User {payload['user_id']} has read permission")
    else:
        print(f"User {payload['user_id']} does not have read permission")
except ValueError as e:
    print(f"Authentication error: {str(e)}")
```

### 8.6.2 敏感操作保护

对于敏感操作，应该实施额外的保护措施。

示例代码：

```python
import hashlib
from typing import Callable, Any

class SensitiveOperationProtector:
    def __init__(self, auth_manager: AuthManager, permission_manager: PermissionManager):
        self.auth_manager = auth_manager
        self.permission_manager = permission_manager

    def protect(self, required_role: str, required_permission: str):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                token = kwargs.get('token')
                if not token:
                    raise ValueError("Authentication token is required")
                
                try:
                    payload = self.auth_manager.verify_token(token)
                    if not self.permission_manager.check_permission(payload['role'], required_permission):
                        raise ValueError(f"User does not have {required_permission} permission")
                    if payload['role'] != required_role:
                        raise ValueError(f"User is not authorized as {required_role}")
                    
                    # 执行操作前记录日志
                    self.log_operation(payload['user_id'], func.__name__)
                    
                    return func(*args, **kwargs)
                except ValueError as e:
                    raise ValueError(f"Authorization failed: {str(e)}")
            return wrapper
        return decorator

    def log_operation(self, user_id: str, operation: str):
        # 在实际应用中，这里应该将日志写入安全的存储系统
        print(f"SENSITIVE OPERATION: User {user_id} performed {operation}")

# 使用示例
protector = SensitiveOperationProtector(auth_manager, permission_manager)

@protector.protect(required_role='admin', required_permission='write')
def delete_user_data(user_id: str, token: str):
    # 执行删除用户数据的操作
    print(f"Deleting data for user {user_id}")

# 尝试执行敏感操作
admin_token = auth_manager.generate_token("admin1", "admin")
try:
    delete_user_data("user123", token=admin_token)
except ValueError as e:
    print(f"Operation failed: {str(e)}")
```

### 8.6.3 审计与合规性保障

实施全面的审计机制以确保系统操作的可追溯性和合规性。

示例代码：

```python
import json
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def get_user_activity(self, user_id: str, start_time: datetime, end_time: datetime):
        user_events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                event_time = datetime.fromisoformat(event['timestamp'])
                if (event['user_id'] == user_id and
                    start_time <= event_time <= end_time):
                    user_events.append(event)
        return user_events

class ComplianceChecker:
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def check_data_access_compliance(self, user_id: str, data_type: str):
        # 检查用户在过去24小时内访问特定类型数据的次数
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        user_events = self.audit_logger.get_user_activity(user_id, start_time, end_time)
        
        access_count = sum(1 for event in user_events 
                           if event['event_type'] == 'data_access' and 
                           event['details'].get('data_type') == data_type)
        
        # 假设我们的合规性规则是：用户每24小时内不能访问同一类型的数据超过10次
        if access_count > 10:
            raise ValueError(f"Compliance violation: User {user_id} has accessed {data_type} data {access_count} times in the last 24 hours")

# 使用示例
audit_logger = AuditLogger("audit_log.jsonl")
compliance_checker = ComplianceChecker(audit_logger)

# 记录数据访问事件
audit_logger.log_event("data_access", "user123", {"data_type": "financial", "record_id": "fin001"})

# 检查合规性
try:
    compliance_checker.check_data_access_compliance("user123", "financial")
    print("Compliance check passed")
except ValueError as e:
    print(f"Compliance violation detected: {str(e)}")
```

这些安全性和权限管理组件共同工作，可以提供一个强大的安全框架：

1. `AuthManager` 处理用户认证，生成和验证 JWT 令牌。
2. `PermissionManager` 管理基于角色的访问控制。
3. `SensitiveOperationProtector` 为敏感操作提供额外的保护层。
4. `AuditLogger` 记录所有重要的系统事件，支持后续的审计和分析。
5. `ComplianceChecker` 实施合规性规则，防止违规操作。

在实际应用中，你可能还需要考虑：

1. 实现更复杂的密码策略，如密码强度要求、定期更改密码等。
2. 添加多因素认证（MFA）以增强安全性。
3. 实现 IP 白名单或地理位置限制，以防止未授权访问。
4. 使用加密技术保护敏感数据，包括传输中和静态数据。
5. 定期进行安全审计和渗透测试，以识别和修复潜在的漏洞。
6. 实现自动化的合规性报告生成功能，以满足各种监管要求。
7. 建立安全事件响应流程，以快速处理潜在的安全威胁。

通过实施这些安全措施，任务自动化 Agent 可以在保护敏感数据和操作的同时，确保系统的可靠性和合规性。这不仅可以防止未授权访问和数据泄露，还可以帮助组织满足各种法规和行业标准的要求。
