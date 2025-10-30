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

import { context } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import * as assert from "assert";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
} from "@google/genai";
import { GoogleGenAIInstrumentation } from "../src/google-genai-instrumentation";

const memoryExporter = new InMemorySpanExporter();

describe("GoogleGenAIInstrumentation", () => {
  const provider = new BasicTracerProvider();
  const instrumentation = new GoogleGenAIInstrumentation({
    traceContent: true,
  });
  let contextManager: AsyncHooksContextManager;

  class FakeModels {
    async generateContent(): Promise<GenerateContentResponse> {
      const response: GenerateContentResponse = {
        modelVersion: "gemini-1.5-pro",
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
        },
        candidates: [
          {
            content: {
              role: "model",
              parts: [{ text: "Hello from Gemini" }],
            },
            finishReason: "STOP",
          },
        ],
      } as unknown as GenerateContentResponse;

      return response;
    }

    async generateContentStream(): Promise<
      AsyncGenerator<GenerateContentResponse>
    > {
      async function* stream() {
        const firstChunk: GenerateContentResponse = {
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "Chunk " }],
              },
            },
          ],
        } as unknown as GenerateContentResponse;
        yield firstChunk;

        const secondChunk: GenerateContentResponse = {
          modelVersion: "gemini-1.5-flash",
          usageMetadata: {
            promptTokenCount: 6,
            candidatesTokenCount: 3,
            totalTokenCount: 9,
          },
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "complete" }],
              },
              finishReason: "STOP",
            },
          ],
        } as unknown as GenerateContentResponse;
        yield secondChunk;
      }

      return stream();
    }
  }

  const fakeModule = {
    Models: FakeModels,
  } as unknown as typeof import("@google/genai");

  before(() => {
    provider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
    instrumentation.setTracerProvider(provider);
    instrumentation.manuallyInstrument(fakeModule);
  });

  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });

  afterEach(() => {
    memoryExporter.reset();
    context.disable();
  });

  it("records attributes for generateContent", async () => {
    const models = new fakeModule.Models();
    await models.generateContent({
      model: "gemini-1.5-pro",
      contents: "Hello there",
      config: {
        temperature: 0.5,
        topP: 0.9,
        maxOutputTokens: 256,
      },
    } satisfies GenerateContentParameters);

    const spans = memoryExporter.getFinishedSpans();
    assert.strictEqual(spans.length, 1);
    const attributes = spans[0].attributes;
    assert.strictEqual(attributes["gen_ai.system"], "Google");
    assert.strictEqual(attributes["gen_ai.request.model"], "gemini-1.5-pro");
    assert.strictEqual(attributes["gen_ai.response.model"], "gemini-1.5-pro");
    assert.strictEqual(attributes["gen_ai.request.temperature"], 0.5);
    assert.strictEqual(attributes["gen_ai.request.top_p"], 0.9);
    assert.strictEqual(attributes["gen_ai.request.max_tokens"], 256);
    assert.strictEqual(attributes["gen_ai.completion.0.role"], "model");
    assert.strictEqual(
      attributes["gen_ai.completion.0.content"],
      "Hello from Gemini",
    );
    assert.strictEqual(attributes["gen_ai.usage.prompt_tokens"], 10);
    assert.strictEqual(attributes["gen_ai.usage.completion_tokens"], 5);
    assert.strictEqual(attributes["llm.usage.total_tokens"], 15);
  });

  it("records attributes for generateContentStream", async () => {
    const models = new fakeModule.Models();
    const stream = await models.generateContentStream({
      model: "gemini-1.5-flash",
      contents: "Streamed request",
    } satisfies GenerateContentParameters);

    const collected: GenerateContentResponse[] = [];
    for await (const chunk of stream) {
      collected.push(chunk);
    }

    assert.strictEqual(collected.length, 2);

    const spans = memoryExporter.getFinishedSpans();
    assert.strictEqual(spans.length, 1);
    const attributes = spans[0].attributes;
    assert.strictEqual(attributes["gen_ai.system"], "Google");
    assert.strictEqual(attributes["gen_ai.request.model"], "gemini-1.5-flash");
    assert.strictEqual(attributes["gen_ai.response.model"], "gemini-1.5-flash");
    assert.strictEqual(attributes["gen_ai.completion.0.role"], "model");
    assert.strictEqual(
      attributes["gen_ai.completion.0.content"],
      "Chunk complete",
    );
    assert.strictEqual(attributes["gen_ai.usage.prompt_tokens"], 6);
    assert.strictEqual(attributes["gen_ai.usage.completion_tokens"], 3);
    assert.strictEqual(attributes["llm.usage.total_tokens"], 9);
  });
});
