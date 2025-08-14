import warnings

warnings.filterwarnings("ignore")

import torch
import time
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box
from rich.columns import Columns
from rich.align import Align

console = Console()


def print_step_header(step_num, title, color="blue"):
    """Beautiful step headers"""
    console.print()
    panel = Panel(
        f"[bold white]STEP {step_num}: {title}[/bold white]",
        style=f"bold {color}",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print(panel)


def visualize_tokens_rich(text, tokenizer):
    """Beautiful token visualization"""
    console.print(f"[bold yellow]📝 Original text:[/bold yellow] '{text}'")

    # Tokenize
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token]) for token in tokens]

    console.print(f"[bold cyan]🔢 Token IDs:[/bold cyan] {tokens}")
    console.print(f"[bold green]📊 Number of tokens:[/bold green] {len(tokens)}")

    # Create beautiful token table
    table = Table(
        title="🔤 Token Breakdown", show_header=True, header_style="bold magenta"
    )
    table.add_column("Index", style="dim", width=8)
    table.add_column("Token ID", justify="right", style="cyan")
    table.add_column("Token String", style="green")
    table.add_column("Visual", style="yellow")

    for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
        # Create visual representation
        visual = f"[bold red]●[/bold red]" if token_str.strip() else "[dim]○[/dim]"
        table.add_row(str(i + 1), str(token_id), f"'{token_str}'", visual)

    console.print(table)


def show_model_info(model, device):
    """Display model information beautifully"""
    info_table = Table(
        title="🤖 Model Information", show_header=True, header_style="bold blue"
    )
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info_table.add_row("Model Name", "GPT-2")
    info_table.add_row("Device", str(device))
    info_table.add_row("Total Parameters", f"{total_params:,}")
    info_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    info_table.add_row("Model Size", f"~{total_params * 4 / 1024**2:.1f} MB")

    console.print(info_table)


def animate_generation():
    """Show generation progress with animation"""
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]Generating text..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=40)
        for i in range(40):
            time.sleep(0.05)  # Simulate generation time
            progress.update(task, advance=1)


def display_final_output(original_text, generated_text, stats):
    """Enhanced final output display with prominent generated text"""

    # Show the complete generated text in a large, prominent panel
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold green]🎉 GENERATION COMPLETE! 🎉[/bold green]",
            style="bold green",
            box=box.DOUBLE,
        )
    )

    # Main output panel - make it really stand out
    console.print("\n")
    output_panel = Panel(
        Text(generated_text, style="bold white", justify="left"),
        title="[bold yellow]📝 COMPLETE GENERATED TEXT[/bold yellow]",
        title_align="center",
        style="bold blue",
        box=box.HEAVY,
        padding=(2, 3),
        width=80,
    )
    console.print(Align.center(output_panel))

    # Separator line for clarity
    console.print("\n" + "─" * 80 + "\n")

    # Show what was originally input vs what was generated
    comparison_table = Table(
        title="📊 Input vs Output Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    comparison_table.add_column("Type", style="yellow", width=15)
    comparison_table.add_column("Text", style="green", width=60)

    # Extract just the new generated part
    original_length = len(tokenizer.encode(original_text))
    new_tokens_only = greedy_output[0][original_length:]
    new_text_only = tokenizer.decode(new_tokens_only, skip_special_tokens=True)

    comparison_table.add_row("Original Input", f"'{original_text}'")
    comparison_table.add_row("AI Generated", f"'{new_text_only}'")
    comparison_table.add_row("Complete Text", f"'{generated_text}'")

    console.print(comparison_table)

    # Stats table
    stats_table = Table(
        title="📈 Generation Statistics", show_header=True, header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="yellow")
    stats_table.add_column("Value", style="green", justify="right")

    for key, value in stats.items():
        stats_table.add_row(key, str(value))

    console.print("\n")
    console.print(stats_table)


def show_generation_process():
    """Show what's happening during generation"""
    process_table = Table(
        title="🧠 AI Thinking Process", show_header=True, header_style="bold red"
    )
    process_table.add_column("Step", style="cyan", width=20)
    process_table.add_column("Description", style="white", width=50)

    steps = [
        ("Attention Analysis", "Looking at relationships between words"),
        ("Context Understanding", "Understanding what 'dog walking' means"),
        ("Next Word Prediction", "Calculating probabilities for 50,257 words"),
        ("Greedy Selection", "Picking the highest probability word"),
        ("Repeat Process", "Doing this 40 times for each new word"),
    ]

    for step, desc in steps:
        process_table.add_row(step, desc)

    console.print(process_table)


# Main execution with beautiful visualization
if __name__ == "__main__":
    # Welcome banner
    console.print(
        Panel.fit(
            "[bold blue]🚀 GPT-2 Text Generation Visualizer 🚀[/bold blue]",
            style="bold blue",
            box=box.DOUBLE,
        )
    )

    print_step_header(1, "INITIALIZING DEVICE AND MODEL", "blue")

    # Device setup with visual feedback
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        console.print("[bold green]✅ GPU (CUDA) detected and activated![/bold green]")
    else:
        torch_device = torch.device("cpu")
        console.print("[bold yellow]⚠️ Using CPU (GPU not available)[/bold yellow]")

    print_step_header(2, "LOADING TOKENIZER AND MODEL", "cyan")

    with console.status("[bold green]Loading GPT-2 components...") as status:
        status.update("[bold blue]📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        status.update("[bold blue]📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", pad_token_id=tokenizer.eos_token_id
        ).to(torch_device)

    console.print("[bold green]✅ All components loaded successfully![/bold green]")

    # Show model info
    show_model_info(model, torch_device)

    print_step_header(3, "TOKENIZATION PROCESS", "magenta")
    input_text = "I enjoy walking with my cute dog"
    visualize_tokens_rich(input_text, tokenizer)

    print_step_header(4, "PREPARING MODEL INPUTS", "yellow")
    model_inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)

    input_info = Table(
        title="📋 Input Information", show_header=True, header_style="bold yellow"
    )
    input_info.add_column("Property", style="cyan")
    input_info.add_column("Value", style="green")

    input_info.add_row("Input Shape", str(tuple(model_inputs["input_ids"].shape)))
    input_info.add_row("Device", str(model_inputs["input_ids"].device))
    input_info.add_row("Data Type", str(model_inputs["input_ids"].dtype))

    console.print(input_info)

    print_step_header(5, "GENERATING TEXT", "red")

    # Show what the AI is doing
    show_generation_process()

    # Animated generation
    animate_generation()

    console.print("\n[bold blue]🤖 Running actual generation...[/bold blue]")
    start_time = time.time()
    greedy_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generation_time = time.time() - start_time

    print_step_header(6, "FINAL RESULTS", "green")

    # Process results
    generated_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    original_length = len(tokenizer.encode(input_text))
    total_length = len(greedy_output[0])
    new_tokens = total_length - original_length
    new_text = tokenizer.decode(
        greedy_output[0][original_length:], skip_special_tokens=True
    )

    # Stats dictionary
    stats = {
        "Generation Time": f"{generation_time:.2f} seconds",
        "Original Tokens": original_length,
        "Generated Tokens": new_tokens,
        "Total Tokens": total_length,
        "Tokens per Second": f"{new_tokens / generation_time:.1f}",
    }

    # Display beautiful final output
    display_final_output(input_text, generated_text, stats)

    # Final closing message
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold green]🎊 Text generation completed successfully! 🎊[/bold green]",
            style="bold green",
            box=box.DOUBLE,
        )
    )

    # Show additional info
    console.print(
        f"\n[bold cyan]💡 The AI continued your sentence by predicting the most likely next words based on patterns learned from millions of texts![/bold cyan]"
    )
