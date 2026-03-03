slint::include_modules!();

use anyhow::Result;
use aphelios_ocr::dolphin::model::DolphinModel;
use slint::{SharedString, ModelRc, VecModel, Model};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::rc::Rc;
use tracing::{info, error};

fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    let window = OcrWindow::new()?;

    // 设置默认模型路径
    window.set_model_path(
        "/Volumes/sw/pretrained_models/Dolphin-v1.5".to_string().into()
    );

    let window_weak = window.as_weak();

    // 绑定选择文件回调
    let window_weak_select = window_weak.clone();
    window.on_select_input_file(move || {
        if let Some(window) = window_weak_select.upgrade() {
            let path = rfd::FileDialog::new()
                .add_filter("图片与 PDF", &["png", "jpg", "jpeg", "pdf", "tiff", "bmp"])
                .pick_file()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            window.set_input_file_path(path.into());
        }
    });

    // 绑定选择目录回调
    let window_weak_dir = window_weak.clone();
    window.on_select_output_dir(move || {
        if let Some(window) = window_weak_dir.upgrade() {
            let path = rfd::FileDialog::new()
                .pick_folder()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            window.set_output_dir_path(path.into());
        }
    });

    // 创建停止标志
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    // 创建日志模型
    let log_model = Rc::new(VecModel::<SharedString>::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    // 绑定开始 OCR 回调
    let window_weak_ocr = window_weak.clone();

    window.on_start_ocr(move || {
        let window_weak_ocr = window_weak_ocr.clone();
        let stop_flag = stop_flag_clone.clone();
        let log_model = log_model.clone();

        // 重置停止标志
        stop_flag.store(false, Ordering::Relaxed);

        // 获取 UI 数据
        let Some(window) = window_weak_ocr.upgrade() else {
            return;
        };

        let input_file = window.get_input_file_path().to_string();
        let output_dir = window.get_output_dir_path().to_string();
        let model_path = window.get_model_path().to_string();

        // 设置 UI 状态
        window.set_is_running(true);
        window.set_status_message("OCR 执行中...".into());

        // 添加初始日志
        log_model.push(format!("开始 OCR: {}", input_file).into());
        log_model.push(format!("输出目录：{}", output_dir).into());
        log_model.push(format!("模型路径：{}", model_path).into());

        // 在新线程中执行 OCR
        std::thread::spawn(move || {
            info!("Starting OCR: input={}, output={}, model={}",
                input_file, output_dir, model_path);

            // 加载模型并执行 OCR
            let result = match DolphinModel::load_model(&model_path) {
                Ok(mut dm) => {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    rt.block_on(dm.dolphin_ocr(&input_file, &output_dir))
                }
                Err(e) => {
                    error!("Failed to load model: {}", e);
                    Err(e)
                }
            };

            // 在后台线程中，我们需要使用 invoke_from_event_loop 来更新 UI
            let window_weak_result = window_weak_ocr.clone();

            match result {
                Ok(results) => {
                    info!("OCR completed with {} results", results.len());
                    let results_clone = results.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(window) = window_weak_result.upgrade() else {
                            return;
                        };
                        let log_model = window.get_log_messages();
                        if let Some(vec_model) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                            vec_model.push(format!("OCR 完成！识别出 {} 条结果", results_clone.len()).into());
                            for (i, result) in results_clone.iter().take(10).enumerate() {
                                vec_model.push(format!("[{}] {}", i + 1, result).into());
                            }
                            if results_clone.len() > 10 {
                                vec_model.push(format!("... 还有 {} 条结果", results_clone.len() - 10).into());
                            }
                        }
                        window.set_status_message("OCR 完成!".into());
                        window.set_is_running(false);
                    });
                }
                Err(e) => {
                    error!("OCR failed: {}", e);
                    let error_msg = e.to_string();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(window) = window_weak_ocr.upgrade() else {
                            return;
                        };
                        let log_model = window.get_log_messages();
                        if let Some(vec_model) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                            vec_model.push(format!("OCR 失败：{}", error_msg).into());
                        }
                        window.set_status_message("OCR 失败".into());
                        window.set_is_running(false);
                    });
                }
            }
        });
    });

    // 绑定停止 OCR 回调
    let window_weak_stop = window_weak.clone();
    window.on_stop_ocr(move || {
        stop_flag.store(true, Ordering::Relaxed);
        if let Some(window) = window_weak_stop.upgrade() {
            window.set_is_running(false);
            window.set_status_message("OCR 已停止".into());
            let log_model = window.get_log_messages();
            if let Some(vec_model) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vec_model.push("OCR 已停止".into());
            }
        }
    });

    window.run()?;

    Ok(())
}
