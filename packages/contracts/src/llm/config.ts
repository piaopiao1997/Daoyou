import { z } from 'zod';

export const LLM_PROVIDER_IDS = ['deepseek', 'alibaba', 'openai'] as const;

export const LlmProviderIdSchema = z.enum(LLM_PROVIDER_IDS);

export type LlmProviderId = z.infer<typeof LlmProviderIdSchema>;

export const LLM_PROVIDER_DEFAULT_MODELS: Record<LlmProviderId, string> = {
  deepseek: 'deepseek-v4-flash',
  alibaba: 'qwen3.7-flash',
  // 自定义 OpenAI 兼容网关没有「官方默认模型」，这里只作为界面占位提示
  openai: 'gpt-4o-mini',
};

export const LlmByokConfigSchema = z
  .object({
    provider: LlmProviderIdSchema,
    apiKey: z.string().trim().min(1).max(512),
    model: z.string().trim().min(1).max(128),
    // 仅「自定义（OpenAI 兼容）」需要：完整的 /v1 地址
    baseUrl: z.url().max(2048).optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.provider === 'openai' && !value.baseUrl) {
      ctx.addIssue({
        code: 'custom',
        path: ['baseUrl'],
        message: '自定义供应商需要填写接口地址',
      });
    }
  });

export type LlmByokConfig = z.infer<typeof LlmByokConfigSchema>;
