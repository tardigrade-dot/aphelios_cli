use crate::cli::{Cli, Commands};

pub mod cli;
pub mod commands;
pub mod engine;
pub mod error;

use crate::error::Result;
use aphelios_ocr::dolphin::model::run_ocr;

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init { path } => {
            commands::init::run(path)?;
        }
        Commands::Base { path, name } => {
            commands::base::run(path, name)?;
        }
        Commands::Run { verbose } => {
            commands::run::run(verbose).await?;
        }
        Commands::QwenLLM { model_id } => {
            commands::qwenvl::run_llm(&model_id).await?;
        }
        Commands::QwenVLM { model_id } => {
            commands::qwenvl::run_vlm(&model_id).await?;
        }
        Commands::Mistral3 { model_id } => {
            commands::mistral3::run_vl(&model_id).await?;
        }
        Commands::Dolphin {
            pdf_path,
            output_path,
        } => {
            run_ocr(&pdf_path, &output_path).await?;
        }
    }
    Ok(())
}
