"""Minimal Gradio app shell for DoRA project demo."""

import gradio as gr


def compare_models(text: str, method: str, rank: int) -> str:
    return (
        f"Input: {text}\n"
        f"Method: {method}\n"
        f"Rank: {rank}\n"
        "Prediction: placeholder\n"
        "Confidence: placeholder"
    )


with gr.Blocks(title="DoRA Final Project Demo") as demo:
    gr.Markdown("# DoRA vs LoRA Demo")
    text = gr.Textbox(label="Input text", lines=4)
    method = gr.Radio(choices=["dora", "lora"], value="dora", label="Method")
    rank = gr.Slider(minimum=2, maximum=16, value=8, step=1, label="Rank")
    output = gr.Textbox(label="Model output", lines=6)
    run_btn = gr.Button("Run")

    run_btn.click(compare_models, inputs=[text, method, rank], outputs=[output])


if __name__ == "__main__":
    demo.launch()
