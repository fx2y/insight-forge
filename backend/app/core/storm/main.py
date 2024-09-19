import asyncio

from langfuse.callback import CallbackHandler

from app.core.storm.graph import storm

langfuse_handler = CallbackHandler()


async def run_storm(topic: str):
    config = {"configurable": {"thread_id": "storm-thread"}, "callbacks": [langfuse_handler]}
    async for step in storm.astream({"topic": topic}, config):
        name = next(iter(step))
        print(f"Completed step: {name}")

    checkpoint = storm.get_state(config)
    return checkpoint.values["article"]


if __name__ == "__main__":
    # topic = "Groq, NVIDIA, Llamma.cpp and the future of LLM Inference"
    # topic = "The potential impact of neuromorphic computing on artificial intelligence"
    topic = "Effectiveness of AI-driven content moderation systems on major social platforms"
    article = asyncio.run(run_storm(topic))
    print(article)
