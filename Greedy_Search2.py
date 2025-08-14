import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
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
from rich.tree import Tree

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
        visual = f"[bold red]●[/bold red]" if token_str.strip() else "[dim]○[/dim]"
        table.add_row(str(i + 1), str(token_id), f"'{token_str}'", visual)

    console.print(table)


def show_probability_step(step_num, current_text, top_candidates, chosen_word):
    """Show probability tree for each generation step"""

    # Create a probability table
    prob_table = Table(
        title=f"🎯 Step {step_num}: Next Word Probabilities",
        show_header=True,
        header_style="bold cyan",
    )
    prob_table.add_column("Rank", style="dim", width=6)
    prob_table.add_column("Word", style="green", width=15)
    prob_table.add_column("Probability", style="yellow", width=12)
    prob_table.add_column("Bar", style="blue", width=20)
    prob_table.add_column("Chosen", style="red", width=8)

    for i, (word, prob) in enumerate(top_candidates):
        # Create visual probability bar
        bar_length = int(prob * 20)  # Scale to 20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # Mark the chosen word - FIXED STYLING
        chosen_mark = "✅" if word == chosen_word else ""

        if word == chosen_word:
            word_display = f"[bold red]'{word}'[/bold red]"
        else:
            word_display = f"'{word}'"

        prob_table.add_row(
            str(i + 1),
            word_display,  # Fixed this line
            f"{prob:.3f}",
            f"[blue]{bar}[/blue]",
            chosen_mark,
        )

    # Show current context
    context_panel = Panel(
        f"[bold white]Current text:[/bold white] \"{current_text}\" → [bold red]'{chosen_word}'[/bold red]",
        title="🔍 Generation Context",
        style="dim",
        box=box.ROUNDED,
    )

    console.print(context_panel)
    console.print(prob_table)
    console.print()


def generate_with_probability_visualization(
    model, tokenizer, input_text, max_new_tokens=10, device="cpu"
):
    """Generate text while showing probability tree at each step"""

    console.print(
        Panel.fit(
            "[bold blue]🌳 PROBABILITY TREE VISUALIZATION[/bold blue]",
            style="bold blue",
        )
    )

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    current_ids = input_ids.clone()
    current_text = input_text

    generated_words = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get model outputs (logits)
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for last token

            # Convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Get top 5 candidates
            top_probs, top_indices = torch.topk(probabilities, 5)

            # Convert to readable format
            top_candidates = []
            for prob, idx in zip(top_probs, top_indices):
                word = tokenizer.decode([idx.item()])
                top_candidates.append((word, prob.item()))

            # Greedy choice (highest probability)
            chosen_token_id = top_indices[0].item()
            chosen_word = tokenizer.decode([chosen_token_id])

            # Show this step's probabilities
            show_probability_step(step + 1, current_text, top_candidates, chosen_word)

            # Update for next iteration
            current_ids = torch.cat(
                [current_ids, torch.tensor([[chosen_token_id]], device=device)], dim=1
            )
            current_text += chosen_word
            generated_words.append(chosen_word)

            # Small delay for dramatic effect
            time.sleep(0.5)

    return current_text, generated_words


def show_final_tree_summary(input_text, generated_words, full_text):
    """Show a tree summary of the generation process"""

    tree = Tree(f"[bold blue]📝 '{input_text}'[/bold blue]")

    current_branch = tree
    accumulated_text = input_text

    for i, word in enumerate(generated_words):
        accumulated_text += word
        word_branch = current_branch.add(
            f"[green]'{word}'[/green] → [dim]\"{accumulated_text}\"[/dim]"
        )
        current_branch = word_branch

    console.print("\n")
    console.print(Panel(tree, title="🌳 Generation Tree Summary", style="bold green"))


# Main execution
if __name__ == "__main__":
    console.print(
        Panel.fit(
            "[bold blue]🚀 GPT-2 Probability Tree Visualizer 🚀[/bold blue]",
            style="bold blue",
            box=box.DOUBLE,
        )
    )

    print_step_header(1, "INITIALIZING MODEL", "blue")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print("[bold green]✅ Using GPU[/bold green]")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]⚠️ Using CPU[/bold yellow]")

    # Load model and tokenizer
    with console.status("[bold green]Loading GPT-2...") as status:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()  # Set to evaluation mode

    console.print("[bold green]✅ Model loaded successfully![/bold green]")

    print_step_header(2, "INPUT TOKENIZATION", "cyan")
    input_text = "I enjoy walking with my cute dog"
    visualize_tokens_rich(input_text, tokenizer)

    print_step_header(3, "PROBABILITY TREE GENERATION", "magenta")

    console.print(
        f"[bold yellow]🎯 Showing top 5 word probabilities at each step[/bold yellow]"
    )
    console.print(
        f"[bold cyan]📊 Greedy search always picks the highest probability word[/bold cyan]"
    )
    console.print()

    # Generate with visualization (showing fewer tokens for clarity)
    full_text, generated_words = generate_with_probability_visualization(
        model, tokenizer, input_text, max_new_tokens=8, device=device
    )

    print_step_header(4, "FINAL RESULTS", "green")

    # Show final output
    output_panel = Panel(
        Text(full_text, style="bold white"),
        title="[bold yellow]📝 COMPLETE GENERATED TEXT[/bold yellow]",
        style="bold blue",
        box=box.HEAVY,
        padding=(1, 2),
    )
    console.print(output_panel)

    # Show generation tree summary
    show_final_tree_summary(input_text, generated_words, full_text)

    # Final stats
    stats_table = Table(
        title="📊 Generation Statistics", show_header=True, header_style="bold cyan"
    )
    stats_table.add_column("Metric", style="yellow")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Original words", str(len(input_text.split())))
    stats_table.add_row("Generated words", str(len(generated_words)))
    stats_table.add_row("Total words", str(len(full_text.split())))
    stats_table.add_row("Generation method", "Greedy Search")

    console.print("\n")
    console.print(stats_table)

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold green]🎊 Probability tree visualization complete! 🎊[/bold green]",
            style="bold green",
        )
    )
