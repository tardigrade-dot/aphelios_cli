use anyhow::Result;
use aphelios_core::utils::logger;

fn main() -> Result<()> {
    logger::init_logging();
    let model_path = "/Volumes/sw/pretrained_models/FireRed-OCR";

    // Qwen3VLModel::new(cfg, vb)
    // let metal: MetalDevice = MetalDevice::new(0)?;
    // let model = VisionModelBuilder::new(model_id)
    //     .with_isq(IsqType::Q4K)
    //     // .with_device(Device::Metal(metal))
    //     // .with_dtype(ModelDType::F16)
    //     .with_logging()
    //     .build()
    //     .await?;

    // let image = image::ImageReader::open("/Users/larry/Documents/resources/car.jpg")?
    //     .decode()
    //     .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    // let messages = VisionMessages::new().add_image_message(
    //     TextMessageRole::User,
    //     "What is this?",
    //     vec![image],
    //     &model,
    // )?;

    // let response = model.send_chat_request(messages).await?;

    // dbg!(response.choices[0].message.content.as_ref().unwrap());

    // dbg!(
    //     response.usage.avg_prompt_tok_per_sec,
    //     response.usage.avg_compl_tok_per_sec
    // );
    Ok(())
}
