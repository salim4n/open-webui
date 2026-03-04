"""
title: Map-Reduce Summarizer
description: A LangGraph-style pipe that summarizes documents using a map-reduce workflow. Short documents get direct summarization; long documents are chunked, summarized in parallel, then consolidated.
author: salim4n
version: 0.1.0
license: MIT
"""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Pipe:
    """Map-Reduce Summarizer Pipe.

    Appears as a selectable model in the chat UI. When selected, orchestrates
    multi-step document summarization with status updates.

    Workflow:
        analyze → route_by_length
            ├─ (short) → direct_summarize → extract_key_points → final_output
            └─ (long)  → chunk_summarize → consolidate → extract_key_points → final_output
    """

    class Valves(BaseModel):
        SUMMARIZATION_MODEL_ID: str = Field(
            default="",
            description="Model ID for LLM calls. Leave empty to auto-select first available non-pipe model.",
        )
        CHUNK_CHAR_THRESHOLD: int = Field(
            default=12000,
            description="Character threshold for short/long routing (~3K tokens).",
        )
        CHUNK_SIZE: int = Field(
            default=10000,
            description="Characters per chunk for the map phase.",
        )
        CHUNK_OVERLAP: int = Field(
            default=500,
            description="Overlap between chunks in characters.",
        )
        MAX_CONCURRENT_CHUNKS: int = Field(
            default=3,
            description="Maximum parallel chunk summarizations.",
        )
        ENABLE_KEY_POINTS: bool = Field(
            default=True,
            description="Whether to extract key points after summarization.",
        )

    class UserValves(BaseModel):
        CUSTOM_INSTRUCTIONS: str = Field(
            default="",
            description="Extra instructions for summarization (focus area, tone, language).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_id(self, body: dict, __request__) -> str:
        """Resolve a real LLM model ID (not this pipe's own ID)."""
        if self.valves.SUMMARIZATION_MODEL_ID:
            return self.valves.SUMMARIZATION_MODEL_ID

        models = getattr(__request__.app.state, "MODELS", {})
        for mid, minfo in models.items():
            if not minfo.get("pipe"):
                return mid

        raise RuntimeError(
            "No available LLM model found. Configure SUMMARIZATION_MODEL_ID in Valves."
        )

    async def _llm_call(
        self,
        __request__,
        __user__: dict,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make a non-streaming LLM call and return the text content."""
        from open_webui.models.users import UserModel
        from open_webui.utils.chat import generate_chat_completion

        user = UserModel(**{k: v for k, v in __user__.items() if k != "valves"})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        form_data = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "metadata": {"skipFilters": True},
        }

        response = await generate_chat_completion(
            __request__,
            form_data=form_data,
            user=user,
            bypass_filter=True,
        )

        # Parse response — may be StreamingResponse or dict
        content = None
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                data = json.loads(chunk.decode("utf-8", "replace"))
                content = data["choices"][0]["message"]["content"]
            if response.background is not None:
                await response.background()
        else:
            content = response["choices"][0]["message"]["content"]

        if content is None:
            raise RuntimeError("LLM returned empty response")
        return content

    async def _emit_status(
        self, __event_emitter__, description: str, done: bool = False
    ):
        """Emit a status event if emitter is available."""
        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    def _extract_document_text(self, body: dict) -> str:
        """Extract document text from the last user message."""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                # Handle multimodal content (list of parts)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts).strip()
        return ""

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        chunk_size = self.valves.CHUNK_SIZE
        overlap = self.valves.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = (
                    current_chunk + "\n\n" + para if current_chunk else para
                )

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _get_custom_instructions(self, __user__: dict) -> str:
        """Get user-specific custom instructions from UserValves."""
        valves = __user__.get("valves")
        if valves and hasattr(valves, "CUSTOM_INSTRUCTIONS"):
            return valves.CUSTOM_INSTRUCTIONS
        return ""

    # ------------------------------------------------------------------
    # Workflow nodes
    # ------------------------------------------------------------------

    async def _direct_summarize(
        self, __request__, __user__: dict, model_id: str, text: str
    ) -> str:
        """Summarize a short document in a single LLM call."""
        custom = self._get_custom_instructions(__user__)
        system = (
            "You are an expert document summarizer. Produce a clear, comprehensive summary "
            "that captures all important information, arguments, and conclusions."
        )
        if custom:
            system += f"\n\nAdditional instructions: {custom}"

        prompt = f"Summarize the following document:\n\n---\n{text}\n---"
        return await self._llm_call(__request__, __user__, model_id, system, prompt)

    async def _summarize_chunk(
        self,
        __request__,
        __user__: dict,
        model_id: str,
        chunk: str,
        index: int,
        total: int,
    ) -> str:
        """Summarize a single chunk (map phase)."""
        custom = self._get_custom_instructions(__user__)
        system = (
            "You are an expert document summarizer. You are summarizing one section of a larger document. "
            "Produce a thorough summary preserving all key information, data points, and arguments."
        )
        if custom:
            system += f"\n\nAdditional instructions: {custom}"

        prompt = (
            f"This is chunk {index + 1} of {total} from a larger document.\n\n"
            f"Summarize this section:\n\n---\n{chunk}\n---"
        )
        return await self._llm_call(__request__, __user__, model_id, system, prompt)

    async def _consolidate(
        self,
        __request__,
        __user__: dict,
        __event_emitter__,
        model_id: str,
        summaries: list[str],
    ) -> str:
        """Consolidate chunk summaries into a single summary (reduce phase).

        If the combined summaries still exceed the threshold, recurse.
        """
        combined = "\n\n---\n\n".join(
            f"**Section {i + 1} Summary:**\n{s}" for i, s in enumerate(summaries)
        )

        # Recursive consolidation if still too long
        if len(combined) > self.valves.CHUNK_CHAR_THRESHOLD:
            await self._emit_status(
                __event_emitter__,
                "Summaries still long, running recursive consolidation...",
            )
            chunks = self._split_into_chunks(combined)
            if len(chunks) > 1:
                sub_summaries = await self._map_chunks(
                    __request__,
                    __user__,
                    __event_emitter__,
                    model_id,
                    chunks,
                    label="consolidation",
                )
                return await self._consolidate(
                    __request__,
                    __user__,
                    __event_emitter__,
                    model_id,
                    sub_summaries,
                )

        custom = self._get_custom_instructions(__user__)
        system = (
            "You are an expert document summarizer. You are given summaries of different sections "
            "of a document. Merge them into a single, coherent, comprehensive summary. "
            "Eliminate redundancy while preserving all unique information."
        )
        if custom:
            system += f"\n\nAdditional instructions: {custom}"

        prompt = (
            "Consolidate these section summaries into one unified summary:\n\n"
            f"{combined}"
        )
        return await self._llm_call(__request__, __user__, model_id, system, prompt)

    async def _extract_key_points(
        self, __request__, __user__: dict, model_id: str, summary: str
    ) -> str:
        """Extract structured key points from the summary."""
        system = (
            "You are an expert analyst. Extract structured insights from the given summary."
        )
        prompt = (
            "From the following summary, extract:\n"
            "1. **Key Points** — the most important takeaways\n"
            "2. **Critical Details** — specific data, numbers, or facts that matter\n"
            "3. **Action Items** — any recommended actions or next steps (if applicable)\n\n"
            f"Summary:\n\n{summary}"
        )
        return await self._llm_call(__request__, __user__, model_id, system, prompt)

    async def _map_chunks(
        self,
        __request__,
        __user__: dict,
        __event_emitter__,
        model_id: str,
        chunks: list[str],
        label: str = "chunk",
    ) -> list[str]:
        """Map phase: summarize chunks in parallel with a semaphore."""
        semaphore = asyncio.Semaphore(self.valves.MAX_CONCURRENT_CHUNKS)
        total = len(chunks)
        results: list[Optional[str]] = [None] * total

        async def process_chunk(index: int):
            async with semaphore:
                await self._emit_status(
                    __event_emitter__,
                    f"Summarizing {label} {index + 1}/{total}...",
                )
                results[index] = await self._summarize_chunk(
                    __request__, __user__, model_id, chunks[index], index, total
                )

        await asyncio.gather(*(process_chunk(i) for i in range(total)))
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Main pipe entry point
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__=None,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:
        """Main pipe function. Yields text chunks for streaming output."""

        if __user__ is None:
            __user__ = {}

        try:
            # --- Analyze ---
            await self._emit_status(__event_emitter__, "Analyzing document...")
            text = self._extract_document_text(body)

            if not text:
                await self._emit_status(
                    __event_emitter__, "Error: No document text found.", done=True
                )
                yield "No document text found. Please provide text to summarize."
                return

            char_count = len(text)
            word_count = len(text.split())
            model_id = self._resolve_model_id(body, __request__)

            await self._emit_status(
                __event_emitter__,
                f"Document: {word_count:,} words, {char_count:,} chars. Model: {model_id}",
            )

            # --- Route ---
            chunks = []
            if char_count <= self.valves.CHUNK_CHAR_THRESHOLD:
                # Short path: direct summarization
                await self._emit_status(
                    __event_emitter__, "Short document — direct summarization..."
                )
                summary = await self._direct_summarize(
                    __request__, __user__, model_id, text
                )
            else:
                # Long path: map-reduce
                chunks = self._split_into_chunks(text)

                if len(chunks) == 1:
                    # Splitting produced a single chunk, use direct path
                    await self._emit_status(
                        __event_emitter__,
                        "Single chunk after splitting — direct summarization...",
                    )
                    summary = await self._direct_summarize(
                        __request__, __user__, model_id, text
                    )
                else:
                    await self._emit_status(
                        __event_emitter__,
                        f"Long document — splitting into {len(chunks)} chunks...",
                    )

                    # Map phase
                    chunk_summaries = await self._map_chunks(
                        __request__,
                        __user__,
                        __event_emitter__,
                        model_id,
                        chunks,
                    )

                    # Reduce phase
                    await self._emit_status(
                        __event_emitter__,
                        "Consolidating chunk summaries...",
                    )
                    summary = await self._consolidate(
                        __request__,
                        __user__,
                        __event_emitter__,
                        model_id,
                        chunk_summaries,
                    )

            # --- Extract key points ---
            key_points = ""
            if self.valves.ENABLE_KEY_POINTS:
                await self._emit_status(
                    __event_emitter__, "Extracting key points..."
                )
                key_points = await self._extract_key_points(
                    __request__, __user__, model_id, summary
                )

            # --- Final output ---
            await self._emit_status(__event_emitter__, "Done.", done=True)

            yield "## Summary\n\n"
            yield summary
            if key_points:
                yield "\n\n---\n\n## Key Points\n\n"
                yield key_points

            yield f"\n\n---\n*Processed {word_count:,} words"
            if len(chunks) > 1:
                yield f" in {len(chunks)} chunks"
            yield f" using `{model_id}`*\n"

        except Exception as e:
            log.exception("Map-Reduce Summarizer error")
            await self._emit_status(
                __event_emitter__, f"Error: {e}", done=True
            )
            yield f"An error occurred during summarization: {e}"
