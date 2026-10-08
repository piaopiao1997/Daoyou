import { getRuntimeEnvironment } from '@server/lib/config/environment.js';
import { createAlibaba } from '@ai-sdk/alibaba';
import { createDeepSeek, deepSeek } from '@ai-sdk/deepseek';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import type { LanguageModel } from 'ai';
import {
  LLM_PROVIDER_DEFAULT_MODELS,
  type LlmProviderId,
} from '@daoyou/contracts/llm/config';

export interface LlmProviderDef {
  id: LlmProviderId;
  defaultModel: string;
  apiKeyEnv: string;
  create: (opts: {
    apiKey?: string;
    baseURL?: string;
    fetch?: typeof fetch;
  }) => (modelId: string) => LanguageModel;
}

const ALIBABA_BASE_URL =
  getRuntimeEnvironment().ALIBABA_BASE_URL?.trim() ||
  'https://dashscope.aliyuncs.com/compatible-mode/v1';

/** 服务端兜底的自定义 OpenAI 兼容地址；BYOK 场景由请求头 `x-llm-base-url` 提供 */
const OPENAI_BASE_URL = getRuntimeEnvironment().OPENAI_BASE_URL?.trim() || '';

export const LLM_PROVIDERS: Record<LlmProviderId, LlmProviderDef> = {
  deepseek: {
    id: 'deepseek',
    defaultModel: LLM_PROVIDER_DEFAULT_MODELS.deepseek,
    apiKeyEnv: 'DEEPSEEK_API_KEY',
    create: ({ apiKey, fetch }) =>
      apiKey || fetch ? createDeepSeek({ apiKey, fetch }) : deepSeek,
  },
  alibaba: {
    id: 'alibaba',
    defaultModel: LLM_PROVIDER_DEFAULT_MODELS.alibaba,
    apiKeyEnv: 'ALIBABA_API_KEY',
    create: ({ apiKey, fetch }) => {
      const provider = createAlibaba({
        apiKey,
        baseURL: ALIBABA_BASE_URL,
        fetch,
      });
      return (modelId: string) => provider(modelId);
    },
  },
  openai: {
    id: 'openai',
    defaultModel: LLM_PROVIDER_DEFAULT_MODELS.openai,
    apiKeyEnv: 'OPENAI_API_KEY',
    create: ({ apiKey, baseURL, fetch }) => {
      const resolvedBaseURL = baseURL?.trim() || OPENAI_BASE_URL;

      if (!resolvedBaseURL) {
        throw new Error(
          '自定义 OpenAI 兼容供应商缺少接口地址：请在设置里填写，或配置服务端 OPENAI_BASE_URL。',
        );
      }

      const provider = createOpenAICompatible({
        name: 'custom-openai',
        apiKey,
        baseURL: resolvedBaseURL,
        // 不开这个，SDK 只会发 {"type":"json_object"}，模型会自己编结构、对不上 Schema
        supportsStructuredOutputs: true,
        fetch,
      });

      return (modelId: string) => provider(modelId);
    },
  },
};
