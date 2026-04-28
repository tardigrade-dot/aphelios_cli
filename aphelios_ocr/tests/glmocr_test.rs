use anyhow::Result;
use aphelios_core::init_logging;
use aphelios_core::utils::common::get_device;
use aphelios_ocr::glmocr::layout::LayoutDetector;
use aphelios_ocr::glmocr::GlmOcr;
use std::time::Instant;

// #[derive(Parser)]
// #[command(name = "glm-ocr")]
// #[command(about = "Pure Rust GLM-OCR inference engine")]
// struct Cli {
//     /// Path to the image file
//     #[arg(short, long)]
//     image: PathBuf,

//     /// OCR prompt (default: "Text Recognition:")
//     #[arg(short, long, default_value = "Text Recognition:")]
//     prompt: String,

//     /// Maximum number of tokens to generate
//     #[arg(short, long, default_value_t = 8192)]
//     max_tokens: usize,

//     /// HuggingFace model ID (default: unsloth/GLM-OCR)
//     #[arg(long)]
//     model_id: Option<String>,

//     /// Use layout detection for intelligent document segmentation
//     #[arg(long)]
//     layout: bool,

//     /// Quantize text decoder weights for faster inference. Levels: q8_0 (default), q4_0 (faster, lower quality)
//     #[arg(long)]
//     quantize: Option<String>,

//     /// Output structured JSON instead of markdown (requires --layout)
//     #[arg(long)]
//     json: bool,

//     /// Use CUDA GPU for inference (requires --features cuda). Specify GPU index, e.g. --gpu 0
//     #[arg(long)]
//     gpu: Option<usize>,
// }

#[test]
fn test_glmocr() -> Result<()> {
    init_logging();

    // Load image
    let image = image::open("/Volumes/sw/MyDrive/data_src/page_zht_49.png")?;

    // Select device
    let device = get_device();

    let layout = false;
    let json = false;
    let max_tokens = 8192;
    let prompt = "Text Recognition:".to_string();

    // Initialize OCR model
    let start = Instant::now();
    let model_id = Some("/Volumes/sw/pretrained_models/GLM-OCR");
    let quantize = Some("q8_0");
    let ocr = GlmOcr::new_with_device(model_id, quantize, device)?;
    let load_time = start.elapsed();
    eprintln!("OCR model loaded in {load_time:.2?}");

    // Run OCR
    let start = Instant::now();
    if layout {
        eprintln!("Loading layout detection model...");
        let layout_start = Instant::now();
        let mut layout = LayoutDetector::new()?;
        eprintln!("Layout model loaded in {:.2?}", layout_start.elapsed());

        if json {
            let doc = ocr.recognize_layout_structured(&image, &mut layout, max_tokens)?;
            let json = doc.to_json()?;
            println!("{json}");
        } else {
            let result = ocr.recognize_with_layout(&image, &mut layout, max_tokens)?;
            println!("{result}");
        }
    } else {
        if json {
            anyhow::bail!("--json requires --layout");
        }
        let result = ocr.recognize_with_max_tokens(&image, &prompt, max_tokens)?;
        println!("{result}");
    };
    let inference_time = start.elapsed();
    eprintln!("Inference completed in {inference_time:.2?}");

    Ok(())
}
