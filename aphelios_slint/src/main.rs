// 包含所有生成的 Slint UI 文件
include!(concat!(env!("OUT_DIR"), "/main_menu.rs"));
include!(concat!(env!("OUT_DIR"), "/ocr_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/asr_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/tts_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/settings_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/book_search_ui.rs"));
include!(concat!(env!("OUT_DIR"), "/text_align_ui.rs"));

mod config;
mod controllers;
mod logger;
mod services;

use anyhow::Result;
use aphelios_core::traits::{OcrEngine, SearchEngine, TtsEngine};
use config::AppSettings;
use controllers::{ocr::OcrLogic, search::SearchLogic, tts::TtsLogic, AppContext};
use slint::{ComponentHandle, Model, ModelRc, PlatformError, SharedString, VecModel};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use tracing::{error, info};

// 日志最大行数限制
const MAX_LOG_LINES: usize = 200;

fn main() -> Result<()> {
    logger::init_slint_logging();

    // 加载配置
    let settings = AppSettings::load().unwrap_or_default();

    // 初始化服务引擎
    let ocr_engine: Arc<Mutex<dyn OcrEngine>> = Arc::new(Mutex::new(services::DolphinOcrClient));
    let search_engine: Arc<dyn SearchEngine> = Arc::new(services::SqliteSearchClient);
    let tts_engine: Arc<dyn TtsEngine> = Arc::new(services::QwenTtsClient);

    // 创建应用上下文
    let ctx = Arc::new(AppContext::new(
        ocr_engine,
        tts_engine,
        search_engine,
        settings,
    ));

    // 设置 macOS 菜单栏 (仅 macOS)
    setup_macos_menu();

    // 显示主菜单
    let main_menu = MainMenu::new()?;
    let main_menu_weak = main_menu.as_weak();

    main_menu.on_open_ocr({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_ocr_ui(ctx.clone());
        }
    });

    main_menu.on_open_asr({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_asr_ui(ctx.clone());
        }
    });

    main_menu.on_open_tts({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_tts_ui(ctx.clone());
        }
    });

    main_menu.on_open_book_search({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_book_search_ui(ctx.clone());
        }
    });

    main_menu.on_open_text_align({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_text_align_ui(ctx.clone());
        }
    });

    main_menu.on_open_settings({
        let w = main_menu_weak.clone();
        let ctx = ctx.clone();
        move || {
            let _ = w.upgrade();
            let _ = run_settings_ui(ctx.clone());
        }
    });

    main_menu.on_quit_app({
        let ctx = ctx.clone();
        move || {
            // 保存配置
            let settings = ctx.get_settings();
            let _ = settings.save();
            std::process::exit(0);
        }
    });

    main_menu.run()?;
    Ok(())
}

/// 设置 macOS 菜单栏
#[cfg(target_os = "macos")]
fn setup_macos_menu() {
    // 使用 slint 的内置菜单支持
    // Slint 会自动在 macOS 上创建应用程序菜单
}

#[cfg(not(target_os = "macos"))]
fn setup_macos_menu() {
    // 非 macOS 平台不做任何操作
}

// ----------------------------------------------------------------------------
// UI 运行逻辑
// ----------------------------------------------------------------------------

fn run_ocr_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = OcrWindow::new()?;

    // 加载保存的设置
    let settings = ctx.get_settings();
    if let Some(model_path) = settings.ocr_model_path {
        window.set_model_path(model_path.into());
    }
    if let Some(output_dir) = settings.ocr_output_dir {
        window.set_output_dir_path(output_dir.into());
    }

    let window_weak = window.as_weak();
    let logic = Arc::new(OcrLogic::new(ctx.clone()));

    window.on_go_back({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存当前设置
                let mut s = ctx.get_settings();
                s.ocr_model_path = Some(win.get_model_path().to_string());
                s.ocr_output_dir = Some(win.get_output_dir_path().to_string());
                ctx.save_settings(s);
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_select_input_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("图片与 PDF", &["png", "jpg", "jpeg", "pdf", "tiff", "bmp"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_input_file_path(path.into());
            }
        }
    });

    window.on_select_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_dir_path(path.into());
            }
        }
    });

    // 使用 Rc 创建 UI 日志模型（UI 线程专用）
    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    // 订阅全局日志
    let mut log_rx = logger::subscribe_logs();
    let w_weak = window.as_weak();
    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.blocking_recv() {
            let _ = slint::invoke_from_event_loop({
                let w = w_weak.clone();
                let msg_clone = msg.clone();
                move || {
                    if let Some(win) = w.upgrade() {
                        let log_model = win.get_log_messages();
                        if let Some(vm) =
                            log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                        {
                            vm.push(msg_clone.into());
                            if vm.row_count() > MAX_LOG_LINES {
                                vm.remove(0);
                            }
                            win.invoke_scroll_to_bottom();
                        }
                    }
                }
            });
        }
    });

    // 添加初始日志
    log_model.push("=== Aphelios OCR 文字识别 ===".into());
    log_model.push("就绪，请选择输入文件开始识别".into());
    window.invoke_scroll_to_bottom();

    window.on_start_ocr({
        let w = window_weak.clone();
        let lm = log_model.clone();
        let logic = logic.clone();
        move || {
            let Some(win) = w.upgrade() else { return };

            let input_file: String = win.get_input_file_path().to_string();
            let output_dir: String = win.get_output_dir_path().to_string();
            let model_path: String = win.get_model_path().to_string();

            win.set_is_running(true);
            win.set_status_message("OCR 执行中...".into());

            // 添加开始日志
            lm.push(format!("📝 开始 OCR: {}", input_file).into());
            lm.push(format!("📂 输出目录：{}", output_dir).into());
            lm.push(format!("🤖 模型路径：{}", model_path).into());
            lm.push("⏳ 正在加载模型并识别...".into());

            let w2 = w.clone();
            logic.start_ocr(
                model_path,
                input_file,
                output_dir,
                lm.clone(),
                move |result| {
                    let w3 = w2.clone(); // Clone inside Fn closure
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3.upgrade() else { return };
                        match result {
                            Ok(results) => {
                                info!("OCR completed with {} results", results.len());
                                let log_model = win.get_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(
                                        format!("✅ OCR 完成！识别出 {} 条结果", results.len())
                                            .into(),
                                    );
                                    for (i, result) in results.iter().take(10).enumerate() {
                                        vm.push(format!("  [{}] {}", i + 1, result).into());
                                    }
                                    if results.len() > 10 {
                                        vm.push(
                                            format!("  ... 还有 {} 条结果", results.len() - 10)
                                                .into(),
                                        );
                                    }
                                    win.invoke_scroll_to_bottom();
                                }
                                win.set_status_message("OCR 完成!".into());
                            }
                            Err(e) => {
                                error!("OCR failed: {}", e);
                                let log_model = win.get_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ OCR 失败：{}", e).into());
                                    win.invoke_scroll_to_bottom();
                                }
                                win.set_status_message("OCR 失败".into());
                            }
                        }
                        win.set_is_running(false);
                    });
                },
            );
        }
    });

    window.on_stop_ocr({
        let w = window_weak.clone();
        let lm = log_model.clone();
        let logic = logic.clone();
        move || {
            logic.stop();
            if let Some(win) = w.upgrade() {
                win.set_is_running(false);
                win.set_status_message("OCR 已停止".into());
                lm.push("⏹️ OCR 已停止".into());
                win.invoke_scroll_to_bottom();
            }
        }
    });

    window.run()?;
    Ok(())
}

fn run_asr_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = AsrWindow::new()?;
    let window_weak = window.as_weak();

    // 加载保存的设置
    let settings = ctx.get_settings();
    if let Some(output_path) = settings.asr_output_path {
        window.set_output_path(output_path.into());
    }

    window.on_go_back({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存设置
                let mut s = ctx.get_settings();
                s.asr_output_path = Some(win.get_output_path().to_string());
                ctx.save_settings(s);
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_select_audio_file({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_audio_file_path(path.into());
            }
        }
    });

    window.on_select_output_path({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_path(path.into());
            }
        }
    });

    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    // 订阅全局日志
    let mut log_rx = logger::subscribe_logs();
    let w_weak = window.as_weak();
    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.blocking_recv() {
            let _ = slint::invoke_from_event_loop({
                let w = w_weak.clone();
                let msg_clone = msg.clone();
                move || {
                    if let Some(win) = w.upgrade() {
                        let log_model = win.get_log_messages();
                        if let Some(vm) =
                            log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                        {
                            vm.push(msg_clone.into());
                            if vm.row_count() > MAX_LOG_LINES {
                                vm.remove(0);
                            }
                            win.invoke_scroll_to_bottom();
                        }
                    }
                }
            });
        }
    });

    // 添加初始日志
    log_model.push("=== Aphelios ASR 语音识别 ===".into());
    log_model.push("就绪，请选择音频文件开始识别".into());
    window.invoke_scroll_to_bottom();

    window.on_start_asr({
        let w: slint::Weak<AsrWindow> = window_weak.clone();
        let lm = log_model.clone();
        let ctx = ctx.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let audio_file: String = win.get_audio_file_path().to_string();
            let output_path: String = win.get_output_path().to_string();

            // 保存设置
            let mut s = ctx.get_settings();
            s.asr_output_path = Some(output_path.clone());
            ctx.save_settings(s);

            win.set_is_running(true);
            win.set_status_message("ASR 识别中...".into());
            lm.push(format!("🎤 开始 ASR 识别：{}", audio_file).into());
            win.invoke_scroll_to_bottom();

            std::thread::spawn(move || {
                info!("Starting ASR: input={}", audio_file);
                // ASR logic currently disabled in original code
            });
        }
    });

    window.run()?;
    Ok(())
}

fn run_settings_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = SettingsWindow::new()?;

    // 加载保存的设置
    let settings = ctx.get_settings();
    if let Some(model_path) = settings.ocr_model_path {
        window.set_ocr_model_path(model_path.into());
    }
    if let Some(output_dir) = settings.ocr_output_dir {
        window.set_ocr_output_dir(output_dir.into());
    }
    if let Some(output_path) = settings.asr_output_path {
        window.set_asr_output_path(output_path.into());
    }
    if let Some(model_path) = settings.srt_model_path {
        window.set_srt_model_path(model_path.into());
    }
    if let Some(min_len) = settings.srt_min_segment_length {
        window.set_srt_min_segment_length(min_len.to_string().into());
    }
    if let Some(max_len) = settings.srt_max_segment_length {
        window.set_srt_max_segment_length(max_len.to_string().into());
    }
    if let Some(model_path) = settings.tts_model_path {
        window.set_tts_model_path(model_path.into());
    }
    if let Some(output_path) = settings.tts_output_path {
        window.set_tts_output_path(output_path.into());
    }
    if let Some(ref_audio) = settings.tts_ref_audio_path {
        window.set_tts_ref_audio_path(ref_audio.into());
    }
    if let Some(ref_text) = settings.tts_ref_text {
        window.set_tts_ref_text(ref_text.into());
    }

    let window_weak = window.as_weak();

    window.on_go_back({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存设置
                let mut s = ctx.get_settings();
                s.ocr_model_path = Some(win.get_ocr_model_path().to_string());
                s.ocr_output_dir = Some(win.get_ocr_output_dir().to_string());
                s.asr_output_path = Some(win.get_asr_output_path().to_string());
                s.srt_model_path = Some(win.get_srt_model_path().to_string());
                s.srt_min_segment_length =
                    Some(win.get_srt_min_segment_length().parse().unwrap_or(80));
                s.srt_max_segment_length =
                    Some(win.get_srt_max_segment_length().parse().unwrap_or(120));
                s.tts_model_path = Some(win.get_tts_model_path().to_string());
                s.tts_output_path = Some(win.get_tts_output_path().to_string());
                s.tts_ref_audio_path = Some(win.get_tts_ref_audio_path().to_string());
                s.tts_ref_text = Some(win.get_tts_ref_text().to_string());
                ctx.save_settings(s);
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_save_settings({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存设置
                let mut s = ctx.get_settings();
                s.ocr_model_path = Some(win.get_ocr_model_path().to_string());
                s.ocr_output_dir = Some(win.get_ocr_output_dir().to_string());
                s.asr_output_path = Some(win.get_asr_output_path().to_string());
                s.srt_model_path = Some(win.get_srt_model_path().to_string());
                s.srt_min_segment_length =
                    Some(win.get_srt_min_segment_length().parse().unwrap_or(80));
                s.srt_max_segment_length =
                    Some(win.get_srt_max_segment_length().parse().unwrap_or(120));
                s.tts_model_path = Some(win.get_tts_model_path().to_string());
                s.tts_output_path = Some(win.get_tts_output_path().to_string());
                s.tts_ref_audio_path = Some(win.get_tts_ref_audio_path().to_string());
                s.tts_ref_text = Some(win.get_tts_ref_text().to_string());
                ctx.save_settings(s);
            }
        }
    });

    window.on_select_ocr_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ocr_model_path(path.into());
            }
        }
    });

    window.on_select_ocr_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ocr_output_dir(path.into());
            }
        }
    });

    window.on_select_asr_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_asr_output_path(path.into());
            }
        }
    });

    window.on_select_srt_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_srt_model_path(path.into());
            }
        }
    });

    window.on_select_tts_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_tts_model_path(path.into());
            }
        }
    });

    window.on_select_tts_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("WAV 音频", &["wav"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_tts_output_path(path.into());
            }
        }
    });

    window.on_select_tts_ref_audio({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_tts_ref_audio_path(path.into());
            }
        }
    });

    window.run()?;
    Ok(())
}

fn run_tts_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = TtsWindow::new()?;

    // 加载保存的设置
    let settings = ctx.get_settings();
    if let Some(model_path) = settings.tts_model_path {
        window.set_model_path(model_path.into());
    }
    if let Some(output_path) = settings.tts_output_path {
        window.set_output_path(output_path.into());
    }
    if let Some(ref_audio) = settings.tts_ref_audio_path {
        window.set_ref_audio_path(ref_audio.into());
    }
    if let Some(ref_text) = settings.tts_ref_text {
        window.set_ref_text(ref_text.into());
    }

    let window_weak = window.as_weak();
    let logic = Arc::new(TtsLogic::new(ctx.clone()));

    window.on_text_changed({
        let w = window.as_weak();
        move |text: slint::SharedString| {
            if let Some(win) = w.upgrade() {
                win.set_input_text_length(text.chars().count() as i32);
            }
        }
    });

    window.on_go_back({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存设置
                let mut s = ctx.get_settings();
                s.tts_model_path = Some(win.get_model_path().to_string());
                s.tts_output_path = Some(win.get_output_path().to_string());
                s.tts_ref_audio_path = Some(win.get_ref_audio_path().to_string());
                s.tts_ref_text = Some(win.get_ref_text().to_string());
                ctx.save_settings(s);
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_select_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_model_path(path.into());
            }
        }
    });

    window.on_select_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("WAV 音频", &["wav"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_path(path.into());
            }
        }
    });

    window.on_select_ref_audio({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ref_audio_path(path.into());
            }
        }
    });

    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    // 订阅全局日志
    let mut log_rx = logger::subscribe_logs();
    let w_weak = window.as_weak();
    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.blocking_recv() {
            let _ = slint::invoke_from_event_loop({
                let w = w_weak.clone();
                let msg_clone = msg.clone();
                move || {
                    if let Some(win) = w.upgrade() {
                        let log_model = win.get_log_messages();
                        if let Some(vm) =
                            log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                        {
                            vm.push(msg_clone.into());
                            if vm.row_count() > MAX_LOG_LINES {
                                vm.remove(0);
                            }
                            win.invoke_scroll_to_bottom();
                        }
                    }
                }
            });
        }
    });

    // 添加初始日志
    log_model.push("=== Aphelios TTS 语音合成 ===".into());
    log_model.push("就绪，请输入文本开始合成".into());
    window.invoke_scroll_to_bottom();

    window.on_start_tts({
        let w = window_weak.clone();
        let lm = log_model.clone();
        let logic = logic.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let input_text: String = win.get_input_text().to_string();
            let model_path: String = win.get_model_path().to_string();
            let output_path: String = win.get_output_path().to_string();
            let ref_audio_path: String = win.get_ref_audio_path().to_string();
            let ref_text: String = win.get_ref_text().to_string();

            win.set_is_running(true);
            win.set_status_message("TTS 合成中...".into());
            win.set_has_audio(false);
            win.set_progress(0.0);
            lm.push(format!("🔊 开始 TTS 合成：{}", input_text).into());
            lm.push(format!("🤖 模型路径：{}", model_path).into());
            lm.push(format!("📂 输出路径：{}", output_path).into());
            win.invoke_scroll_to_bottom();

            let w2 = w.clone();
            let logic_inner = logic.clone();
            logic.start_tts(
                input_text,
                model_path,
                output_path.clone(),
                ref_audio_path,
                ref_text,
                {
                    let w_progress = w2.clone();
                    move |p| {
                        let _ = slint::invoke_from_event_loop({
                            let w = w_progress.clone();
                            move || {
                                if let Some(win) = w.upgrade() {
                                    win.set_progress(p);
                                }
                            }
                        });
                    }
                },
                move |result| {
                    let output_path_inner = output_path.clone();
                    let w3 = w2.clone();
                    let logic_final = logic_inner.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3.upgrade() else { return };
                        win.set_is_running(false);
                        match result {
                            Ok(_) => {
                                info!("TTS completed: {}", output_path_inner);
                                win.set_status_message("TTS 完成!".into());
                                win.set_has_audio(true);
                                let log_model = win.get_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push("✅ TTS 合成完成!".into());
                                    win.invoke_scroll_to_bottom();
                                }
                                if let Ok(mut path) = logic_final.ctx().audio_output_path.lock() {
                                    *path = Some(output_path_inner);
                                }
                            }
                            Err(e) => {
                                error!("TTS failed: {}", e);
                                win.set_status_message("TTS 失败".into());
                                let log_model = win.get_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ TTS 失败：{}", e).into());
                                    win.invoke_scroll_to_bottom();
                                }
                            }
                        }
                    });
                },
            );
        }
    });

    window.on_play_audio({
        let w = window_weak.clone();
        let lm = log_model.clone();
        let ctx = ctx.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            if let Ok(path_guard) = ctx.audio_output_path.lock() {
                if let Some(ref path) = *path_guard {
                    lm.push(format!("🔊 播放音频：{}", path).into());
                    win.invoke_scroll_to_bottom();
                    let path_clone = path.clone();
                    std::thread::spawn(move || {
                        play_audio_file(&path_clone);
                    });
                }
            }
        }
    });

    window.run()?;
    Ok(())
}

fn run_book_search_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = BookSearchWindow::new()?;
    let logic = Arc::new(SearchLogic::new(ctx.clone()));

    // 获取书籍数量
    let book_count = logic.get_book_count().unwrap_or(0);
    window.set_book_count(book_count as i32);

    let window_weak = window.as_weak();

    window.on_go_back({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_build_index({
        let w = window_weak.clone();
        let logic = logic.clone();
        move || {
            let Some(win) = w.upgrade() else { return };

            win.set_is_indexing(true);
            win.set_status_message("正在构建索引...".into());

            let w2 = w.clone();
            logic.build_index(move |result| {
                let w3 = w2.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(win) = w3.upgrade() else { return };
                    win.set_is_indexing(false);
                    match result {
                        Ok(count) => {
                            win.set_book_count(count as i32);
                            win.set_status_message(
                                format!("索引构建完成，共 {} 本书", count).into(),
                            );
                        }
                        Err(e) => {
                            win.set_status_message(format!("索引构建失败: {}", e).into());
                        }
                    }
                });
            });
        }
    });

    window.on_search_books({
        let w = window_weak.clone();
        let logic = logic.clone();
        move |query: slint::SharedString| {
            let Some(win) = w.upgrade() else { return };

            win.set_status_message("搜索中...".into());

            let w2 = w.clone();
            logic.search_books(query.to_string(), 50, move |result| {
                let w3 = w2.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(win) = w3.upgrade() else { return };
                    match result {
                        Ok(search_result) => {
                            let items = search_result.books;
                            let book_items: Vec<slint_generatedBookSearchWindow::BookItem> = items
                                .iter()
                                .map(|b| slint_generatedBookSearchWindow::BookItem {
                                    id: b.id.to_string().into(),
                                    title: b.title.clone().into(),
                                    author: b.author.clone().unwrap_or_default().into(),
                                    file_path: b.file_path.clone().into(),
                                    file_type: b.file_type.clone().into(),
                                    file_size: format_file_size(b.file_size).into(),
                                })
                                .collect();

                            let model: Rc<VecModel<slint_generatedBookSearchWindow::BookItem>> =
                                Rc::new(VecModel::from(book_items));
                            win.set_search_results(ModelRc::from(model));
                            win.set_status_message(
                                format!("找到 {} 个结果", search_result.total).into(),
                            );
                        }
                        Err(e) => {
                            win.set_status_message(format!("搜索失败: {}", e).into());
                        }
                    }
                });
            });
        }
    });

    window.on_open_book({
        move |file_path: slint::SharedString| {
            let path: String = file_path.into();
            if !path.is_empty() {
                // 使用 open 命令打开文件
                std::process::Command::new("open").arg(&path).spawn().ok();
            }
        }
    });

    window.run()?;
    Ok(())
}

fn format_file_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if size >= GB {
        format!("{:.2} GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.2} MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.2} KB", size as f64 / KB as f64)
    } else {
        format!("{} B", size)
    }
}

fn play_audio_file(path: &str) {
    // Audio playback logic currently disabled or placeholder
    info!("Playing audio file: {}", path);
}

// ----------------------------------------------------------------------------
// 文本对齐 UI 运行逻辑
// ----------------------------------------------------------------------------

fn run_text_align_ui(ctx: Arc<AppContext>) -> Result<()> {
    let window = TextAlignWindow::new()?;
    let window_weak = window.as_weak();

    // 加载保存的设置
    let settings = ctx.get_settings();
    if let Some(model_path) = settings.srt_model_path {
        window.set_model_path(model_path.into());
    }
    if let Some(min_len) = settings.srt_min_segment_length {
        window.set_min_segment_length(min_len.to_string().into());
    }
    if let Some(max_len) = settings.srt_max_segment_length {
        window.set_max_segment_length(max_len.to_string().into());
    }

    window.on_go_back({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                // 保存设置
                let mut s = ctx.get_settings();
                s.srt_model_path = Some(win.get_model_path().to_string());
                s.srt_min_segment_length = Some(win.get_min_segment_length().parse().unwrap_or(80));
                s.srt_max_segment_length =
                    Some(win.get_max_segment_length().parse().unwrap_or(120));
                ctx.save_settings(s);
                let _: Result<(), PlatformError> = win.hide();
            }
        }
    });

    window.on_select_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_model_path(path.into());
            }
        }
    });

    window.on_select_audio_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let audio_path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                if !audio_path.is_empty() {
                    win.set_audio_file_path(audio_path.clone().into());

                    // 自动设置文本文件路径（同目录同名.txt）
                    let audio_path_buf = std::path::PathBuf::from(&audio_path);
                    let txt_path = audio_path_buf.with_extension("txt");
                    win.set_target_text_path(txt_path.to_string_lossy().to_string().into());

                    // 自动设置输出 SRT 路径（同目录同名.srt）
                    let srt_path = audio_path_buf.with_extension("srt");
                    win.set_output_path(srt_path.to_string_lossy().to_string().into());
                }
            }
        }
    });

    window.on_select_target_text({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_target_text_path(path.into());
            }
        }
    });

    window.on_select_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("SRT 文件", &["srt"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_output_path(path.into());
            }
        }
    });

    let log_model: Rc<VecModel<SharedString>> = Rc::new(VecModel::default());
    window.set_log_messages(ModelRc::from(log_model.clone()));

    // 订阅全局日志
    let mut log_rx = logger::subscribe_logs();
    let w_weak = window.as_weak();
    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.blocking_recv() {
            let _ = slint::invoke_from_event_loop({
                let w = w_weak.clone();
                let msg_clone = msg.clone();
                move || {
                    if let Some(win) = w.upgrade() {
                        let log_model = win.get_log_messages();
                        if let Some(vm) =
                            log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                        {
                            vm.push(msg_clone.into());
                            if vm.row_count() > MAX_LOG_LINES {
                                vm.remove(0);
                            }
                            win.invoke_scroll_to_bottom();
                        }
                    }
                }
            });
        }
    });

    // 添加初始日志
    log_model.push("=== Aphelios 文本对齐 - 生成 SRT 字幕 ===".into());
    log_model.push("就绪，请选择音频和目标文本开始对齐".into());
    window.invoke_scroll_to_bottom();

    window.on_start_alignment({
        let w = window_weak.clone();
        let lm = log_model.clone();
        move || {
            let Some(win) = w.upgrade() else { return };

            let model_path: String = win.get_model_path().to_string();
            let audio_file: String = win.get_audio_file_path().to_string();
            let target_text: String = win.get_target_text_path().to_string();
            let output_path: String = win.get_output_path().to_string();
            let min_len: i32 = win.get_min_segment_length().parse().unwrap_or(80);
            let max_len: i32 = win.get_max_segment_length().parse().unwrap_or(120);

            win.set_is_running(true);
            win.set_status_message("正在对齐...".into());
            lm.push(format!("🎤 开始文本对齐").into());
            lm.push(format!("🤖 模型：{}", model_path).into());
            lm.push(format!("🎵 音频：{}", audio_file).into());
            lm.push(format!("📄 文本：{}", target_text).into());
            lm.push(format!("💾 输出：{}", output_path).into());
            lm.push(format!("⚙️ 分段长度：{} ~ {} 字", min_len, max_len).into());
            win.invoke_scroll_to_bottom();

            let w2 = w.clone();
            let output_path_clone = output_path.clone();
            std::thread::spawn(move || {
                // 执行文本对齐逻辑
                let result = aphelios_asr::text_match::audio_text_match_with_params(
                    &model_path,
                    &audio_file,
                    &target_text,
                    if output_path_clone.is_empty() {
                        None
                    } else {
                        Some(&output_path_clone)
                    },
                    Some(min_len as usize),
                    Some(max_len as usize),
                );

                let _ = slint::invoke_from_event_loop(move || {
                    let Some(win) = w2.upgrade() else { return };
                    win.set_is_running(false);

                    match result {
                        Ok(srt_path) => {
                            info!("文本对齐完成：{}", srt_path);
                            let lm = win.get_log_messages();
                            if let Some(vm) = lm.as_any().downcast_ref::<VecModel<SharedString>>() {
                                vm.push(
                                    format!("✅ 对齐完成！SRT 文件已保存到：{}", srt_path).into(),
                                );
                            }
                            win.set_status_message("对齐完成!".into());

                            // 读取 SRT 文件内容用于预览
                            if let Ok(content) = std::fs::read_to_string(&srt_path) {
                                win.set_srt_preview(content.into());
                                win.invoke_scroll_preview_to_bottom();
                            }
                        }
                        Err(e) => {
                            error!("文本对齐失败：{}", e);
                            let lm = win.get_log_messages();
                            if let Some(vm) = lm.as_any().downcast_ref::<VecModel<SharedString>>() {
                                vm.push(format!("❌ 对齐失败：{}", e).into());
                            }
                            win.set_status_message("对齐失败".into());
                        }
                    }
                    win.invoke_scroll_to_bottom();
                });
            });
        }
    });

    window.on_stop_alignment({
        let w = window_weak.clone();
        let lm = log_model.clone();
        move || {
            // 目前无法中断正在进行的 ASR 识别
            if let Some(win) = w.upgrade() {
                win.set_is_running(false);
                win.set_status_message("已停止".into());
                lm.push("⏹️ 已停止".into());
                win.invoke_scroll_to_bottom();
            }
        }
    });

    window.run()?;
    Ok(())
}
