use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mycli")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init {
        path: String,
    },
    Base {
        path: String,
        name: String,
    },
    Run {
        #[arg(short, long)]
        verbose: bool,
    },
    QwenLLM {
        model_id: String,
    },
    QwenVLM {
        model_id: String,
    },
    Mistral3 {
        model_id: String,
    },
    Dolphin {
        pdf_path: String,
        output_path: String,
    },
}
