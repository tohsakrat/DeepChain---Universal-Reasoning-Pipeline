"""
title: DeepChain
author: your_name
description: DeepChain 思维链处理管道 - Open WebUI 0.5.6+
version: 1.0.1
licence: MIT
project_url: https://github.com/tohsakrat/DeepChain---Universal-Reasoning-Pipeline
author_url: https://github.com/tohsakrat
required_open_webui_version: 0.5.6
❤❤❤ "If you find this project helpful, please consider giving it a star! ❤❤❤

作为open-webui的pipeline function，嫁接reasoning model的思考链，引导其他任何兼容openai接口的语言模型的输出。

说明和标题都是ai生成的，如不恰当请轻踹↓

该工具实现推理模型与生成模型的智能协作，主要功能：  
1. 支持DeepSeek标准（通过reasoning_content字段）和OpenAI标准（使用<think>标签）的思考链  
2. 兼容DeepSeek官方/OpenRouter接口（DeepSeek标准）及Nebius等OpenAI标准服务  
3. 为任意兼容OpenAI的模型提供结构化推理引导

**DeepClaude Pipeline** bridges reasoning models with any OpenAI-compatible LLMs. Key features:  

1. **Chain-of-Thought Integration**  
   - Supports DeepSeek-style reasoning (via `reasoning_content` field) and OpenAI-style CoT (using `<think>` tags)  
   - Outputs standardized OpenAI format with `<think>` wrapped reasoning  

2. **Universal Compatibility**  
   - Works with DeepSeek official/OpenRouter APIs (DeepSeek standard)  
   - Compatible with OpenAI-style services like Nebius  
   - Guides any OpenAI-compatible model using reasoning context  

3. **Optimized Processing**  
   - Automatically injects reasoning chains into prompts  
   - Maintains clean output formatting  

"""

import json
import httpx
import re
import logging
from typing import AsyncGenerator, Callable, Awaitable, Optional
from pydantic import BaseModel, Field
import asyncio

log = logging.getLogger(__name__)


class UnifiedDelta(BaseModel):
    reasoning: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[str] = None


class Pipe:
    class Valves(BaseModel):
        REASONING_API_BASE: str = Field(
            default="https://api.deepseek.com/v1", description="推理模型API地址"
        )
        REASONING_API_KEY: str = Field(default="", description="推理模型API密钥")
        REASONING_MODEL: str = Field(  # 新增推理模型参数
            default="deepseek-reasoner",
            description="推理阶段使用模型（示例：deepseek-reasoner/gpt-4-turbo）",
        )
        GENERATION_API_BASE: str = Field(
            default="https://api.anthropic.com/v1", description="生成模型API地址"
        )
        GENERATION_API_KEY: str = Field(default="", description="生成模型API密钥")
        GENERATION_MODEL: str = Field(  # 新增生成模型参数
            default="claude-3-5-sonnet-20240620",
            description="生成阶段使用模型（示例：claude-3.5-sonnet/gpt-4o）",
        )
        MODEL_ALIAS: str = Field(
            default="CoT-Pipeline", description="管道显示名称（前端可见名称）"
        )
        CONTENT_TYPE: str = Field(
            default="auto",
            description="推理模型数据格式检测模式：\n"
            "auto - 自动检测接口类型\n"
            "deepseek - 强制使用DeepSeek格式\n"
            "openai - 强制使用OpenAI格式",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.state = "init"  # init → reasoning → generating → done
        self.buffer = []
        self.emitter = None
        self.final_reason = ""

    def pipes(self):
        return [
            {
                "id": "cot-pipeline",
                "name": self.valves.MODEL_ALIAS,
            }
        ]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__
        self._clean_historical_messages(body["messages"])
        log.info(f"Received request: {json.dumps(body, indent=2)}")

        try:
            # 阶段1：处理思维链
            self.state = "generation"
            async for chunk in self._process_reasoning(body):
                yield chunk
            # 阶段2：生成最终响应
            self.state = "responsing"
            async for chunk in self._generate_response(body):
                yield chunk

        except Exception as e:
            error_chunk = {"error": f"{type(e).__name__}: {str(e)}"}
            log.error(f"Pipeline error: {error_chunk}")
            yield json.dumps(error_chunk, ensure_ascii=False)

    def _enhance_messages(self, original_messages: list) -> list:
        """增强输入消息（通用版本），基于思维链获得引导词"""
        if not original_messages:
            return

        # 深拷贝避免污染原始数据
        processed_messages = [msg.copy() for msg in original_messages]

        # 处理最后一条用户消息
        final_user_msg = next(
            (msg for msg in reversed(processed_messages) if msg["role"] == "user"), None
        )

        if final_user_msg:
            # 构建增强内容
            original_input = final_user_msg.get("content", "")

            enhanced_content = (
                f"## 用户原始输入\n{original_input}\n\n"
                f"## 你的思考过程\n{self.final_reason }\n\n"
                "请基于以上思考给出最终答复："
            )
            final_user_msg["content"] = enhanced_content

        # 过滤系统级消息并返回
        return [
            msg
            for msg in processed_messages
            if msg.get("role", "") not in {"system", "tool"}
        ]

    def _clean_historical_messages(self, messages: list):
        """清理历史消息中的推理痕迹"""
        pattern = re.compile(
            r'<details\s+type="reasoning"[^>]*>.*?</details>', re.DOTALL
        )
        for msg in messages:
            if "content" in msg:
                msg["content"] = pattern.sub("", msg["content"]).strip()

    async def _process_reasoning(self, body: dict) -> AsyncGenerator[str, None]:
        """处理思维链阶段"""
        client = httpx.AsyncClient(http2=True)
        self.current_request = client
        # self.buffer = []
        try:
            # 发起推理请求
            if self.valves.CONTENT_TYPE == "deepseek":
                yield "<think>\n"
            # yield f"{self.valves.GENERATION_API_BASE}/chat/completions"

            payload: dict = {
                **body,
                "model": self.valves.REASONING_MODEL,
                "stream": True,
                "include_reasoning": True,
            }

            async with client.stream(
                "POST",
                f"{self.valves. REASONING_API_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.valves.REASONING_API_KEY}"},
                timeout=30,
            ) as response:

                if response.status_code != 200:
                    error = response.json().get("error", {})

                    raise RuntimeError(
                        f"Reasoning API error: {error.get('message', 'Unknown')}"
                    )

                # 进入流处理
                self.state = "reasoning"

                # yield json.dumps(body, ensure_ascii=False)  # 输出参数调试log不好用的时候用

                async for line in response.aiter_lines():
                    # log.info(f"Received raw data: {line}")
                    # yield "{" + json.dumps(line, ensure_ascii=False) + "}" + "\n" + "\n"

                    if (not line) or (not line.startswith("data: ")):
                        continue
                    # yield "进入请求" + line + "\n"
                    data = json.loads(line[6:])
                    # yield "读取到数据" + json.dumps(data, ensure_ascii=False) + "\n"
                    # yield json.dumps(data, ensure_ascii=False) + "\n"
                    delta = self._parse_delta(data)
                    # yield json.dumps(delta, ensure_ascii=False) + "\n"
                    # if self._should_end_reasoning(data):
                    #    yield "True"
                    # else:
                    #    yield "False"

                    if self._should_end_reasoning(data):
                        break
                    reasoning_content = ""
                    if self.valves.CONTENT_TYPE == "deepseek" and delta.get(
                        "reasoning_content"
                    ):
                        # yield "deepseek模式"

                        # yield self._should_end_reasoning(data)
                        reasoning_content = delta["reasoning_content"]
                    else:
                        # yield "openai模式"
                        # yield json.dumps(delta, ensure_ascii=False) + "\n"

                        reasoning_content = delta["content"]
                        # yield json.dumps(delta)
                        # yield json.dumps(delta, ensure_ascii=False)
                        # yield delta.get("content")
                    if not reasoning_content:
                        continue
                    yield reasoning_content
                    self.buffer.append(reasoning_content)
                # 终止请求以节省资源
                await response.aclose()

            # 发送缓冲内容
            self.final_reason = "".join(self.buffer)
            # if self.buffer:
            # yield self.final_reason
            yield "\n</think>\n"

        finally:
            await client.aclose()

    async def _generate_response(
        self, body: dict
    ) -> AsyncGenerator[UnifiedDelta, None]:
        """生成最终响应"""
        enhanced_body = {**body, "messages": self._enhance_messages(body["messages"])}

        config = {
            "base_url": self.valves.GENERATION_API_BASE,
            "headers": {"Authorization": f"Bearer {self.valves.GENERATION_API_KEY}"},
            "endpoint": "/chat/completions",
            "model": self.valves.GENERATION_MODEL,
        }

        async with httpx.AsyncClient(http2=True) as client:
            payload = {**enhanced_body, "model": config["model"], "stream": True}
            async with client.stream(
                "POST",
                f"{config['base_url']}{config['endpoint']}",
                json=payload,
                headers=config["headers"],
                timeout=300,
            ) as response:
                # yield f"{config['base_url']}{config['endpoint']}" + "\n"
                # yield json.dumps(payload, ensure_ascii=False)
                if response.status_code != 200:
                    error = await response.aread()
                    raise RuntimeError(
                        f"API Error ({response.status_code}): {error.decode()[:200]}"
                    )

                async for line in response.aiter_lines():
                    # yield f"Raw generation data: {line}")
                    # yield json.dumps(line, ensure_ascii=False) + "\n"
                    if (not line) or not (line.startswith("data: ")):
                        continue

                    data = json.loads(line[6:])
                    delta = self._parse_delta(data)
                    # yield json.dumps(data, ensure_ascii=False) + "\n"
                    # yield json.dumps(delta, ensure_ascii=False) + "\n"
                    if data.get("choices", [{}])[0].get("finish_reason"):
                        return
                    if delta["content"]:
                        yield delta["content"]

    def _parse_delta(self, data: dict) -> dict:
        """解析不同格式的delta数据"""
        delta = {}
        content = ""
        if self.valves.CONTENT_TYPE == "deepseek":
            content = (
                data.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
            )

            if not content:
                content = data.get("choices", [{}])[0].get("delta", {}).get("reasoning")

            if not content:
                content = data.get("choices", [{}])[0].get("delta", {}).get("content")

        else:
            content = data.get("choices", [{}])[0].get("delta", {}).get("content")
        delta["reasoning_content"] = content
        delta["content"] = content

        return delta

    def _should_end_reasoning(self, data: dict) -> bool:
        """判断是否结束思维链"""
        # return True
        if self.valves.CONTENT_TYPE == "deepseek":
            # DeepSeek标准：出现content字段且没有reasoning_content
            return (
                bool(data.get("choices", [{}])[0].get("delta", {}).get("content"))
                and not data.get("choices", [{}])[0]
                .get("delta", {})
                .get("reasoning_content")
                and not data.get("choices", [{}])[0].get("delta", {}).get("reasoning")
            )

        else:
            # OpenAI标准：检测到标签

            return bool(
                data.get("choices", [{}])[0].get("delta", {}).get("content")
                == "</think>"
            )
