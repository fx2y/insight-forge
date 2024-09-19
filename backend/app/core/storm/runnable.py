import json
from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable

from app.core.storm.chain import expand_chain, gen_perspectives_chain, gen_queries_chain, gen_answer_chain
from app.core.storm.constant import MAX_STR_LEN
from app.core.storm.helper import format_docs, swap_roles, tag_with_name
from app.core.storm.llm import fast_llm, long_context_llm
from app.core.storm.model import InterviewState, WikiSection
from app.core.storm.prompt import gen_qn_prompt, section_writer_prompt
from app.core.storm.tool import search_engine
from app.core.storm.util import wikipedia_retriever, retriever


@as_runnable
async def survey_subjects(topic: str):
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
            RunnableLambda(swap_roles).bind(name=editor.name)
            | gen_qn_prompt.partial(persona=editor.persona)
            | fast_llm
            | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


async def gen_answer(
        state: InterviewState,
        config: Optional[dict] = None,
        name: str = "Subject_Matter_Expert",
        max_str_len: int = MAX_STR_LEN,
):
    swapped_state = swap_roles(state, name)
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}


async def retrieve(inputs: dict):
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    return {"docs": formatted, **inputs}


section_writer = (
        retrieve
        | section_writer_prompt
        | long_context_llm.with_structured_output(WikiSection)
)
