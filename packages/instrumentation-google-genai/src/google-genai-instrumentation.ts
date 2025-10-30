/*
 * Copyright Traceloop
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  Attributes,
  Span,
  SpanKind,
  SpanStatusCode,
  context,
  trace,
} from "@opentelemetry/api";
import {
  InstrumentationBase,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import type * as googleGenAI from "@google/genai";
import { GoogleGenAIInstrumentationConfig } from "./types";
import { version } from "../package.json";

const MODULE_NAME = "@google/genai";
const DEFAULT_SPAN_NAME = "google.genai.completion";

type GenerateContentParameters = googleGenAI.GenerateContentParameters;
type GenerateContentResponse = googleGenAI.GenerateContentResponse;
type GenerateContentResponseUsageMetadata =
  googleGenAI.GenerateContentResponseUsageMetadata;
type Models = googleGenAI.Models;
type Candidate = googleGenAI.Candidate;
type Content = googleGenAI.Content;
type ContentListUnion = googleGenAI.ContentListUnion;
type ContentUnion = googleGenAI.ContentUnion;
type Part = googleGenAI.Part;
type PartUnion = googleGenAI.PartUnion;

type AsyncResponseGenerator = AsyncGenerator<GenerateContentResponse>;

interface CompletionAggregation {
  role?: string;
  parts: string[];
  finishReason?: string;
}

interface StreamingAggregation {
  completions: Map<number, CompletionAggregation>;
  usageMetadata?: GenerateContentResponseUsageMetadata;
  modelVersion?: string;
}

export class GoogleGenAIInstrumentation extends InstrumentationBase {
  declare protected _config: GoogleGenAIInstrumentationConfig;

  constructor(config: GoogleGenAIInstrumentationConfig = {}) {
    super("@traceloop/instrumentation-google-genai", version, config);
  }

  public override setConfig(config: GoogleGenAIInstrumentationConfig = {}) {
    super.setConfig(config);
  }

  protected init(): InstrumentationModuleDefinition {
    return new InstrumentationNodeModuleDefinition(
      MODULE_NAME,
      [">=1.0.0"],
      this.wrap.bind(this),
      this.unwrap.bind(this),
    );
  }

  public manuallyInstrument(module: typeof googleGenAI) {
    this._diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.applyPatches(module);
  }

  private wrap(module: typeof googleGenAI, moduleVersion?: string) {
    this._diag.debug(`Patching ${MODULE_NAME}@${moduleVersion}`);
    this.applyPatches(module);

    const defaultExport = (module as unknown as { default?: typeof googleGenAI })
      .default;
    if (defaultExport && defaultExport !== module) {
      this.applyPatches(defaultExport);
    }

    return module;
  }

  private unwrap(module: typeof googleGenAI, moduleVersion?: string): void {
    this._diag.debug(`Unpatching ${MODULE_NAME}@${moduleVersion}`);
    this.removePatches(module);

    const defaultExport = (module as unknown as { default?: typeof googleGenAI })
      .default;
    if (defaultExport && defaultExport !== module) {
      this.removePatches(defaultExport);
    }
  }

  private applyPatches(exports: Partial<typeof googleGenAI>) {
    if (!exports?.Models?.prototype) {
      this._diag.debug(
        "Google GenAI instrumentation: Models prototype not found, skipping patch.",
      );
      return;
    }

    this._wrap(
      exports.Models.prototype,
      "generateContent",
      this.createGenerateContentWrapper(),
    );
    this._wrap(
      exports.Models.prototype,
      "generateContentStream",
      this.createGenerateContentStreamWrapper(),
    );
  }

  private removePatches(exports: Partial<typeof googleGenAI>) {
    if (!exports?.Models?.prototype) {
      return;
    }

    this._unwrap(exports.Models.prototype, "generateContent");
    this._unwrap(exports.Models.prototype, "generateContentStream");
  }

  private createGenerateContentWrapper() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const plugin = this;
    return function wrapGenerateContent(
      original: (
        params: GenerateContentParameters,
      ) => Promise<GenerateContentResponse>,
    ) {
      return function wrappedGenerateContent(
        this: Models,
        params: GenerateContentParameters,
      ) {
        const span = plugin.startSpan(params);
        const execContext = trace.setSpan(context.active(), span);

        const execPromise = safeExecuteInTheMiddle(
          () =>
            context.with(execContext, () => original.apply(this, [params])),
          (error) => {
            if (error) {
              plugin._diag.error(
                "Error in Google GenAI instrumentation",
                error,
              );
            }
          },
        ) as Promise<GenerateContentResponse>;

        const wrappedPromise = plugin.wrapResponsePromise(span, execPromise);
        return context.bind(execContext, wrappedPromise);
      };
    };
  }

  private createGenerateContentStreamWrapper() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const plugin = this;
    return function wrapGenerateContentStream(
      original: (
        params: GenerateContentParameters,
      ) => Promise<AsyncResponseGenerator>,
    ) {
      return function wrappedGenerateContentStream(
        this: Models,
        params: GenerateContentParameters,
      ) {
        const span = plugin.startSpan(params);
        const execContext = trace.setSpan(context.active(), span);

        const execPromise = safeExecuteInTheMiddle(
          () =>
            context.with(execContext, () => original.apply(this, [params])),
          (error) => {
            if (error) {
              plugin._diag.error(
                "Error in Google GenAI streaming instrumentation",
                error,
              );
            }
          },
        ) as Promise<AsyncResponseGenerator>;

        const wrappedPromise = plugin.wrapStreamPromise(span, execPromise);
        return context.bind(execContext, wrappedPromise);
      };
    };
  }

  private startSpan(params: GenerateContentParameters): Span {
    const attributes: Attributes = {
      [SpanAttributes.LLM_SYSTEM]: "Google",
      [SpanAttributes.LLM_REQUEST_TYPE]: LLMRequestTypeValues.COMPLETION,
    };

    try {
      if (params.model) {
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = params.model;
        attributes[SpanAttributes.LLM_RESPONSE_MODEL] = params.model;
      }

      if (params.config) {
        const config = params.config;
        if (config.maxOutputTokens !== undefined) {
          attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] =
            config.maxOutputTokens;
        }
        if (config.temperature !== undefined) {
          attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] =
            config.temperature;
        }
        if (config.topP !== undefined) {
          attributes[SpanAttributes.LLM_REQUEST_TOP_P] = config.topP;
        }
        if (config.topK !== undefined) {
          attributes[SpanAttributes.LLM_TOP_K] = config.topK;
        }
        if (config.frequencyPenalty !== undefined) {
          attributes[SpanAttributes.LLM_FREQUENCY_PENALTY] =
            config.frequencyPenalty;
        }
        if (config.presencePenalty !== undefined) {
          attributes[SpanAttributes.LLM_PRESENCE_PENALTY] =
            config.presencePenalty;
        }
        if (config.stopSequences?.length) {
          attributes[SpanAttributes.LLM_CHAT_STOP_SEQUENCES] = JSON.stringify(
            config.stopSequences,
          );
        }

        if (this.shouldSendPrompts()) {
          const systemContent = this.normalizeContentUnion(config.systemInstruction);
          if (systemContent) {
            const systemFormatted = this.formatPartsData(systemContent.parts);
            if (systemFormatted) {
              attributes[`${SpanAttributes.LLM_PROMPTS}.0.role`] =
                systemContent.role ?? "system";
              attributes[`${SpanAttributes.LLM_PROMPTS}.0.content`] =
                systemFormatted;
            }
          }
        }
      }

      if (this.shouldSendPrompts()) {
        const contents = this.normalizeContentList(params.contents);
        const promptOffset = params.config?.systemInstruction ? 1 : 0;
        contents.forEach((content, index) => {
          const formatted = this.formatPartsData(content.parts);
          if (formatted) {
            attributes[
              `${SpanAttributes.LLM_PROMPTS}.${promptOffset + index}.role`
            ] = content.role ?? "user";
            attributes[
              `${SpanAttributes.LLM_PROMPTS}.${promptOffset + index}.content`
            ] = formatted;
          }
        });
      }
    } catch (error) {
      this._diag.debug(
        "Failed to collect Google GenAI request attributes",
        error as Error,
      );
      this._config.exceptionLogger?.(error as Error);
    }

    return this.tracer.startSpan(DEFAULT_SPAN_NAME, {
      kind: SpanKind.CLIENT,
      attributes,
    });
  }

  private wrapResponsePromise(
    span: Span,
    promise: Promise<GenerateContentResponse>,
  ): Promise<GenerateContentResponse> {
    return promise
      .then((result) => {
        this.applyResponseAttributes(span, result);
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return result;
      })
      .catch((error: Error) => {
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        span.recordException(error);
        span.end();
        throw error;
      });
  }

  private wrapStreamPromise(
    span: Span,
    promise: Promise<AsyncResponseGenerator>,
  ): Promise<AsyncResponseGenerator> {
    return promise
      .then((generator) => this.wrapAsyncGenerator(span, generator))
      .catch((error: Error) => {
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        span.recordException(error);
        span.end();
        throw error;
      });
  }

  private wrapAsyncGenerator(span: Span, generator: AsyncResponseGenerator) {
    const aggregation: StreamingAggregation = {
      completions: new Map<number, CompletionAggregation>(),
    };
    let spanEnded = false;
    const finalize = () => {
      if (spanEnded) {
        return;
      }
      spanEnded = true;
      try {
        this.applyStreamingAttributes(span, aggregation);
        span.setStatus({ code: SpanStatusCode.OK });
      } catch (error) {
        this._diag.debug(
          "Failed to finalize Google GenAI streaming span",
          error as Error,
        );
        this._config.exceptionLogger?.(error as Error);
      } finally {
        span.end();
      }
    };

    const processChunk = (chunk?: GenerateContentResponse) => {
      if (!chunk) {
        return;
      }

      try {
        if (chunk.modelVersion) {
          aggregation.modelVersion = chunk.modelVersion;
        }
        if (chunk.usageMetadata) {
          aggregation.usageMetadata = chunk.usageMetadata;
        }
        if (chunk.candidates) {
          chunk.candidates.forEach((candidate, index) => {
            const existing =
              aggregation.completions.get(index) ?? this.createEmptyCompletion();
            if (candidate.content?.role && !existing.role) {
              existing.role = candidate.content.role;
            }
            if (candidate.finishReason) {
              existing.finishReason = candidate.finishReason;
            }
            const formatted = this.formatPartsData(candidate.content?.parts);
            if (formatted) {
              existing.parts.push(formatted);
            }
            aggregation.completions.set(index, existing);
          });
        }
      } catch (error) {
        this._diag.debug(
          "Failed to aggregate Google GenAI streaming chunk",
          error as Error,
        );
        this._config.exceptionLogger?.(error as Error);
      }
    };

    const wrappedIterator: AsyncResponseGenerator = {
      next: async (...args: Parameters<AsyncResponseGenerator["next"]>) => {
        try {
          const result = await generator.next(...args);
          processChunk(result.value);
          if (result.done) {
            finalize();
          }
          return result;
        } catch (error) {
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: (error as Error).message,
          });
          span.recordException(error as Error);
          finalize();
          throw error;
        }
      },
      return: async (...args: Parameters<AsyncResponseGenerator["return"]>) => {
        try {
          const [value] = args;
          processChunk(value);
          const result = generator.return
            ? await generator.return(...args)
            : { done: true, value };
          finalize();
          return result;
        } catch (error) {
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: (error as Error).message,
          });
          span.recordException(error as Error);
          finalize();
          throw error;
        }
      },
      throw: async (...args: Parameters<AsyncResponseGenerator["throw"]>) => {
        try {
          const result = generator.throw
            ? await generator.throw(...args)
            : Promise.reject(args[0]);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: args[0] instanceof Error ? args[0].message : undefined,
          });
          if (args[0] instanceof Error) {
            span.recordException(args[0]);
          }
          finalize();
          return result;
        } catch (error) {
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: (error as Error).message,
          });
          span.recordException(error as Error);
          finalize();
          throw error;
        }
      },
      [Symbol.asyncIterator]() {
        return this;
      },
      async [Symbol.asyncDispose]() {
        finalize();
      },
    };

    return wrappedIterator;
  }

  private applyResponseAttributes(span: Span, response: GenerateContentResponse) {
    try {
      if (response.modelVersion) {
        span.setAttribute(
          SpanAttributes.LLM_RESPONSE_MODEL,
          response.modelVersion,
        );
      }

      if (response.usageMetadata) {
        this.applyUsageMetadata(span, response.usageMetadata);
      }

      if (response.candidates) {
        const completions = response.candidates.map((candidate, index) => ({
          index,
          role: candidate.content?.role ?? "assistant",
          content: this.formatPartsData(candidate.content?.parts),
          finishReason: candidate.finishReason,
        }));
        this.applyCompletions(span, completions);
      }
    } catch (error) {
      this._diag.debug(
        "Failed to collect Google GenAI response attributes",
        error as Error,
      );
      this._config.exceptionLogger?.(error as Error);
    }
  }

  private applyStreamingAttributes(
    span: Span,
    aggregation: StreamingAggregation,
  ) {
    if (aggregation.modelVersion) {
      span.setAttribute(
        SpanAttributes.LLM_RESPONSE_MODEL,
        aggregation.modelVersion,
      );
    }

    if (aggregation.usageMetadata) {
      this.applyUsageMetadata(span, aggregation.usageMetadata);
    }

    if (!this.shouldSendPrompts()) {
      return;
    }

    Array.from(aggregation.completions.entries()).forEach(
      ([index, completion]) => {
        const role = completion.role ?? "assistant";
        if (role) {
          span.setAttribute(
            `${SpanAttributes.LLM_COMPLETIONS}.${index}.role`,
            role,
          );
        }
        if (completion.finishReason) {
          span.setAttribute(
            `${SpanAttributes.LLM_COMPLETIONS}.${index}.finish_reason`,
            completion.finishReason,
          );
        }
        if (completion.parts.length > 0) {
          span.setAttribute(
            `${SpanAttributes.LLM_COMPLETIONS}.${index}.content`,
            completion.parts.join(""),
          );
        }
      },
    );
  }

  private applyUsageMetadata(
    span: Span,
    usage: GenerateContentResponseUsageMetadata,
  ) {
    if (usage.totalTokenCount !== undefined) {
      span.setAttribute(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        usage.totalTokenCount,
      );
    }
    if (usage.candidatesTokenCount !== undefined) {
      span.setAttribute(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        usage.candidatesTokenCount,
      );
    }
    if (usage.promptTokenCount !== undefined) {
      span.setAttribute(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        usage.promptTokenCount,
      );
    }
  }

  private applyCompletions(
    span: Span,
    completions: Array<{
      index: number;
      role?: string;
      content?: string;
      finishReason?: Candidate["finishReason"];
    }>,
  ) {
    if (!this.shouldSendPrompts()) {
      return;
    }

    completions.forEach((completion) => {
      if (completion.role) {
        span.setAttribute(
          `${SpanAttributes.LLM_COMPLETIONS}.${completion.index}.role`,
          completion.role,
        );
      }
      if (completion.finishReason) {
        span.setAttribute(
          `${SpanAttributes.LLM_COMPLETIONS}.${completion.index}.finish_reason`,
          completion.finishReason,
        );
      }
      if (completion.content) {
        span.setAttribute(
          `${SpanAttributes.LLM_COMPLETIONS}.${completion.index}.content`,
          completion.content,
        );
      }
    });
  }

  private normalizeContentList(contents: ContentListUnion): Content[] {
    if (Array.isArray(contents)) {
      if (contents.length === 0) {
        return [];
      }

      if (this.isContent(contents[0])) {
        return (contents as Content[]).map((content) => ({
          role: content.role,
          parts: content.parts,
        }));
      }

      return [
        {
          role: "user",
          parts: (contents as PartUnion[]).map((part) => this.normalizePart(part)),
        },
      ];
    }

    if (typeof contents === "string") {
      return [
        {
          role: "user",
          parts: [{ text: contents }],
        },
      ];
    }

    if (this.isPart(contents)) {
      return [
        {
          role: "user",
          parts: [this.normalizePart(contents)],
        },
      ];
    }

    return [contents as Content];
  }

  private normalizeContentUnion(content?: ContentUnion) {
    if (!content) {
      return undefined;
    }

    if (Array.isArray(content)) {
      return {
        role: "system",
        parts: content.map((part) => this.normalizePart(part)),
      };
    }

    if (typeof content === "string") {
      return {
        role: "system",
        parts: [{ text: content }],
      };
    }

    if (this.isPart(content)) {
      return {
        role: "system",
        parts: [this.normalizePart(content)],
      };
    }

    return content as Content;
  }

  private normalizePart(part: PartUnion): Part {
    if (typeof part === "string") {
      return { text: part };
    }
    return part;
  }

  private isContent(value: unknown): value is Content {
    return (
      typeof value === "object" &&
      value !== null &&
      ("parts" in (value as Record<string, unknown>) ||
        "role" in (value as Record<string, unknown>))
    );
  }

  private isPart(value: unknown): value is Part {
    if (typeof value !== "object" || value === null) {
      return false;
    }

    const candidate = value as Record<string, unknown>;
    return (
      "text" in candidate ||
      "inlineData" in candidate ||
      "fileData" in candidate ||
      "functionCall" in candidate ||
      "functionResponse" in candidate ||
      "codeExecutionResult" in candidate ||
      "executableCode" in candidate
    );
  }

  private formatPartsData(parts?: Part[]): string {
    if (!parts?.length) {
      return "";
    }

    return parts
      .map((part) => {
        if (part.text) {
          return part.text;
        }
        if (part.fileData) {
          const uri = part.fileData.fileUri ?? "";
          const mimeType = part.fileData.mimeType ?? "";
          return `${uri}-${mimeType}`;
        }
        if (part.inlineData) {
          const mimeType = part.inlineData.mimeType ?? "";
          const data = part.inlineData.data ?? "";
          return `${data}-${mimeType}`;
        }
        if (part.functionCall) {
          return JSON.stringify(part.functionCall);
        }
        if (part.functionResponse) {
          return JSON.stringify(part.functionResponse);
        }
        if (part.codeExecutionResult) {
          return JSON.stringify(part.codeExecutionResult);
        }
        if (part.executableCode) {
          return JSON.stringify(part.executableCode);
        }
        return "";
      })
      .filter((value) => value !== "")
      .join("\n");
  }

  private shouldSendPrompts() {
    const contextShouldSendPrompts = context
      .active()
      .getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) as boolean | undefined;

    if (contextShouldSendPrompts !== undefined) {
      return contextShouldSendPrompts;
    }

    return this._config.traceContent ?? true;
  }

  private createEmptyCompletion(): CompletionAggregation {
    return {
      role: undefined,
      finishReason: undefined,
      parts: [],
    };
  }
}
