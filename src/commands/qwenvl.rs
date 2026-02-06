use anyhow::{Result,Error as E};
use candle_core::{Device, MetalDevice};
use candle_core::backend::BackendDevice;
use mistralrs::{ModelDType, IsqType, TextMessageRole, TextMessages, TextModelBuilder, VisionMessages, VisionModelBuilder};
use tracing::info;

use crate::measure_time;


pub async fn run_vlm(model_id: &str) -> Result<()> {
    
    let metal: MetalDevice = MetalDevice::new(0)?;
    let model = VisionModelBuilder::new(model_id)
        .with_isq(IsqType::Q4K)
        // .with_device(Device::Metal(metal))
        // .with_dtype(ModelDType::F16)
        .with_logging()
        .build()
        .await?;

    let image = image::ImageReader::open("/Users/larry/Documents/resources/car.jpg")?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is this?",
        vec![image],
        &model,
    )?;

    let response = model.send_chat_request(messages).await?;

    dbg!(response.choices[0].message.content.as_ref().unwrap());

    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}

pub async fn book_search(model_id: &str) -> Result<String>{

    let book_list = "1 - 哈耶克作品法律、立法与自由(全三册) ([英]弗里德利希·冯·哈 耶克 [耶克, 弗里德利希·冯·哈]) (Z-Library).epub
2 - 過去與未來之間：政治思考的八場習練 (漢娜．鄂蘭(Hannah Arendt)) (Z-Library).epub
3 - 本真性的伦理 ([加]查尔斯·泰勒，程炼译) (Z-Library).pdf
4 - 西奥迪尼社会心理学 群体与社会如何影响自我 原书第5版 (（美）道格拉斯·肯里克，史蒂文·纽伯格，罗伯特·西奥迪尼著；谢晓非等译) (Z-Library).pdf
5 - 1905 ([俄] 列夫·达维多维奇·托洛茨基 著, Uni 佐仓绫奈 译校) (Z-Library).pdf
6 - 自由与权力 ([英]阿克顿 [未知]) (Z-Library).epub
7 - 普通法 (【美】小奥利弗·温德尔·霍姆斯) (Z-Library).epub
8 - 自然语言处理实战 ( etc.) (Z-Library).epub
9 - 血色大地：夹在希特勒与史达林之间的东欧 = Bloodlands Europe Between Hitler and Stalin (提摩希 · 史奈德（Timothy Snyder）著；陈荣彬, 刘维人 译) (Z-Library)(t2c).epub";

    let metal: MetalDevice = MetalDevice::new(0)?;
    let model = TextModelBuilder::new(model_id)
        // .with_isq(IsqType::Q4K)
        .with_logging()
        .with_dtype(ModelDType::F16)
        .with_device(Device::Metal(metal))
        .build()
        .await?;

    let mut messages = TextMessages::new();
    messages = messages.add_message(TextMessageRole::User, book_list);
    messages = messages.add_message(TextMessageRole::User, "查找与东欧相关的书籍, 返回名称前的编号和书籍名称");
    let response = model.send_chat_request(messages.clone()).await?;

    let res = response.choices[0].message.content.as_ref().unwrap();
    info!("{}", res);

    Ok(res.clone())
}

pub async fn run_llm(model_id: &str) -> Result<()>{

    let metal: MetalDevice = MetalDevice::new(0)?;

    let model = measure_time!("loading model", TextModelBuilder::new(model_id)
            .with_isq(IsqType::Q4K)
            .with_logging()
            .with_dtype(ModelDType::F16)
            .with_device(Device::Metal(metal))
            .build()
            .await?);

    let mut messages = TextMessages::new();
    messages = messages.add_message(TextMessageRole::User, "介绍下博弈论");
    let response = measure_time!("first chat", model.send_chat_request(messages.clone()).await?);

    info!("{}", response.choices[0].message.content.as_ref().unwrap());

    messages = messages.add_message(
        TextMessageRole::Assistant,
        response.choices[0].message.content.as_ref().unwrap(),
    );

    // ------------------------------------------------------------------
    // Second question, disable thinking mode with RequestBuilder or /no_think
    // ------------------------------------------------------------------
    messages = messages.add_message(TextMessageRole::User, "天空为什么是蓝色的?");

    let response = measure_time!("second chat", model.send_chat_request(messages.clone()).await?);

    info!("{}", response.choices[0].message.content.as_ref().unwrap());

    messages = messages.add_message(
        TextMessageRole::Assistant,
        response.choices[0].message.content.as_ref().unwrap(),
    );

    // ------------------------------------------------------------------
    // Third question, reenable thinking mode with RequestBuilder or /think
    // ------------------------------------------------------------------
    messages = messages.add_message(TextMessageRole::User, "介绍下存在主义 /think");
    let response = model.send_chat_request(messages).await?;

    info!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
