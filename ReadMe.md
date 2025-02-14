
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
