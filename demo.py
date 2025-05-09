# demo.py
import gradio as gr
from quake_talk.search import semantic_search
from quake_talk.gpt    import ask
import polars as pl

def chat_fn(question, temp):
    idxs, _ = semantic_search(question, top_k=5)
    df = pl.read_parquet("artifacts/tweets_cleaned.parquet")
    context = "\n".join(df["content_clean"].take(idxs).to_list())
    return ask(question, context, temperature=temp)

demo = gr.Interface(
    fn=chat_fn,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Temperature")
    ],
    outputs="text",
    title="Quake Talk Demo"
)
demo.launch()
