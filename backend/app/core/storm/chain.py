from langchain_core.output_parsers import StrOutputParser

from app.core.storm.llm import fast_llm, long_context_llm
from app.core.storm.model import Outline, RelatedSubjects, Perspectives, Queries, AnswerWithCitations
from app.core.storm.prompt import direct_gen_outline_prompt, gen_related_topics_prompt, gen_perspectives_prompt, \
    gen_queries_prompt, gen_answer_prompt, refine_outline_prompt, writer_prompt

generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(Outline)

expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(RelatedSubjects)

gen_perspectives_chain = gen_perspectives_prompt | fast_llm.with_structured_output(Perspectives)

gen_queries_chain = gen_queries_prompt | fast_llm.with_structured_output(Queries, include_raw=True)

gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
    AnswerWithCitations, include_raw=True
).with_config(run_name="GenerateAnswer")

refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(Outline)

writer = writer_prompt | long_context_llm | StrOutputParser()
