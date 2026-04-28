// 包含生成的 Slint UI 文件（app_window.slint 自动导入所有页面组件）
include!(concat!(env!("OUT_DIR"), "/app_window.rs"));

mod config;
mod controllers;
mod services;

use anyhow::Result;
use aphelios_core::traits::{OcrEngine, SearchEngine, SearchMode, TtsEngine};
use aphelios_core::utils::logger;
use config::AppSettings;
use controllers::{
    demucs::DemucsLogic, ocr::OcrLogic, search::SearchLogic, tts::TtsLogic, AppContext,
};
use slint::{CloseRequestResponse, ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use tracing::{error, info};

use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    Icon,
};

#[cfg(target_os = "macos")]
use objc2::MainThreadMarker;
#[cfg(target_os = "macos")]
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

// 日志最大行数限制
const MAX_LOG_LINES: usize = 200;

fn main() -> Result<()> {
    logger::init_slint_logging();
    initialize_desktop_app();

    // 加载配置
    let settings = AppSettings::load().unwrap_or_default();

    // 初始化服务引擎
    let ocr_engine: Arc<Mutex<dyn OcrEngine>> = Arc::new(Mutex::new(services::DolphinOcrClient));
    let book_dir = settings.books_dir.as_deref().unwrap_or("/Volumes/sw/books");
    let search_client = Arc::new(services::InMemorySearchClient::new(book_dir));
    let search_engine: Arc<dyn SearchEngine> = search_client.clone();
    let search_engine_for_rescan = search_engine.clone();
    let tts_engine: Arc<dyn TtsEngine> = Arc::new(services::QwenTtsClient);

    // 创建应用上下文
    let ctx = Arc::new(AppContext::new(
        ocr_engine,
        tts_engine,
        search_engine,
        settings,
    ));

    // ── 创建主窗口 ──
    let window = AppWindow::new()?;
    window.set_app_version(env!("CARGO_PKG_VERSION").into());
    window
        .window()
        .on_close_requested(|| CloseRequestResponse::HideWindow);

    let window_weak = window.as_weak();

    // ── 从设置加载各页面配置 ──
    {
        let s = ctx.get_settings();

        // OCR
        if let Some(ref path) = s.ocr_model_path {
            window.set_ocr_model_path(path.clone().into());
        }
        if let Some(ref dir) = s.ocr_output_dir {
            window.set_ocr_output_dir_path(dir.clone().into());
        }

        // ASR
        if let Some(ref path) = s.asr_model_path {
            window.set_asr_model_path(path.clone().into());
        }
        if let Some(ref path) = s.asr_aligner_model_path {
            window.set_asr_aligner_model_path(path.clone().into());
        }
        if let Some(ref path) = s.asr_vad_model_path {
            window.set_asr_vad_model_path(path.clone().into());
        }

        // TTS
        if let Some(ref path) = s.tts_model_path {
            window.set_tts_model_path(path.clone().into());
        }
        if let Some(ref path) = s.tts_output_path {
            window.set_tts_output_path(path.clone().into());
        }
        if let Some(ref path) = s.tts_ref_audio_path {
            window.set_tts_ref_audio_path(path.clone().into());
        }
        if let Some(ref text) = s.tts_ref_text {
            window.set_tts_ref_text(text.clone().into());
        }

        // 文本对齐 (SRT)
        if let Some(ref path) = s.srt_model_path {
            window.set_ta_model_path(path.clone().into());
        }
        if let Some(ref min_len) = s.srt_min_segment_length {
            window.set_ta_min_segment_length(min_len.to_string().into());
        }
        if let Some(ref max_len) = s.srt_max_segment_length {
            window.set_ta_max_segment_length(max_len.to_string().into());
        }

        // Demucs
        if let Some(ref path) = s.demucs_model_path {
            window.set_demucs_model_path(path.clone().into());
        }

        // 设置页面
        if let Some(path) = s.ocr_model_path {
            window.set_settings_ocr_model_path(path.into());
        }
        if let Some(dir) = s.ocr_output_dir {
            window.set_settings_ocr_output_dir(dir.into());
        }
        if let Some(path) = s.asr_model_path {
            window.set_settings_asr_model_path(path.into());
        }
        if let Some(path) = s.asr_aligner_model_path {
            window.set_settings_asr_aligner_model_path(path.into());
        }
        if let Some(path) = s.asr_vad_model_path {
            window.set_settings_asr_vad_model_path(path.into());
        }
        if let Some(path) = s.srt_model_path {
            window.set_settings_srt_model_path(path.into());
        }
        if let Some(min_len) = s.srt_min_segment_length {
            window.set_settings_srt_min_segment_length(min_len.to_string().into());
        }
        if let Some(max_len) = s.srt_max_segment_length {
            window.set_settings_srt_max_segment_length(max_len.to_string().into());
        }
        if let Some(path) = s.tts_model_path {
            window.set_settings_tts_model_path(path.into());
        }
        if let Some(path) = s.tts_output_path {
            window.set_settings_tts_output_path(path.into());
        }
        if let Some(path) = s.tts_ref_audio_path {
            window.set_settings_tts_ref_audio_path(path.into());
        }
        if let Some(text) = s.tts_ref_text {
            window.set_settings_tts_ref_text(text.into());
        }
        if let Some(path) = s.demucs_model_path {
            window.set_settings_demucs_model_path(path.into());
        }
        if let Some(dir) = s.books_dir {
            window.set_settings_books_dir(dir.into());
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // OCR 页面回调
    // ═══════════════════════════════════════════════════════════════
    let logic_ocr = Arc::new(OcrLogic::new(ctx.clone()));

    window.on_ocr_select_input_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("图片与 PDF", &["png", "jpg", "jpeg", "pdf", "tiff", "bmp"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ocr_input_file_path(path.into());
            }
        }
    });

    window.on_ocr_select_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ocr_output_dir_path(path.into());
            }
        }
    });

    window.on_ocr_select_model_path({
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

    window.on_ocr_start_ocr({
        let w = window_weak.clone();
        let logic = logic_ocr.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let input_file: String = win.get_ocr_input_file_path().to_string();
            let output_dir: String = win.get_ocr_output_dir_path().to_string();
            let model_path: String = win.get_ocr_model_path().to_string();

            win.set_ocr_is_running(true);
            win.set_ocr_status_message("OCR 执行中...".into());

            let log_model = win.get_ocr_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push(format!("📝 开始 OCR: {}", input_file).into());
                vm.push(format!("📂 输出目录：{}", output_dir).into());
                vm.push(format!("🤖 模型路径：{}", model_path).into());
                vm.push("⏳ 正在加载模型并识别...".into());
            }

            let w2 = w.clone();
            logic.start_ocr(
                model_path,
                input_file,
                output_dir,
                Rc::new(VecModel::default()),
                move |result| {
                    let w3 = w2.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3.upgrade() else { return };
                        match result {
                            Ok(results) => {
                                info!("OCR completed with {} results", results.len());
                                let log_model = win.get_ocr_log_messages();
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
                                    win.invoke_ocr_scroll_to_bottom();
                                }
                                win.set_ocr_status_message("OCR 完成!".into());
                            }
                            Err(e) => {
                                error!("OCR failed: {}", e);
                                let log_model = win.get_ocr_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ OCR 失败：{}", e).into());
                                    win.invoke_ocr_scroll_to_bottom();
                                }
                                win.set_ocr_status_message("OCR 失败".into());
                            }
                        }
                        win.set_ocr_is_running(false);
                    });
                },
            );
        }
    });

    window.on_ocr_stop_ocr({
        let w = window_weak.clone();
        let logic = logic_ocr.clone();
        move || {
            logic.stop();
            if let Some(win) = w.upgrade() {
                win.set_ocr_is_running(false);
                win.set_ocr_status_message("OCR 已停止".into());
                let log_model = win.get_ocr_log_messages();
                if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                    vm.push("⏹️ OCR 已停止".into());
                    win.invoke_ocr_scroll_to_bottom();
                }
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // ASR 页面回调
    // ═══════════════════════════════════════════════════════════════
    window.on_asr_select_audio_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_asr_audio_file_path(path.clone().into());
                if !path.is_empty() {
                    let audio_path_buf = std::path::PathBuf::from(&path);
                    let srt_path = audio_path_buf.with_extension("srt");
                    win.set_asr_output_path(srt_path.to_string_lossy().to_string().into());
                }
            }
        }
    });

    window.on_asr_select_output_path({
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

    window.on_asr_start_asr({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let asr_model: String = win.get_asr_model_path().to_string();
            let aligner_model: String = win.get_asr_aligner_model_path().to_string();
            let vad_model: String = win.get_asr_vad_model_path().to_string();
            let audio_file: String = win.get_asr_audio_file_path().to_string();
            let output_path: String = win.get_asr_output_path().to_string();
            let language: String = win.get_asr_language().to_string();

            let mut s = ctx.get_settings();
            s.asr_model_path = Some(asr_model.clone());
            s.asr_aligner_model_path = Some(aligner_model.clone());
            s.asr_vad_model_path = Some(vad_model.clone());
            ctx.save_settings(s);

            win.set_asr_is_running(true);
            win.set_asr_progress(0.3);
            win.set_asr_status_message("ASR 识别中...".into());

            let log_model = win.get_asr_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push("=== Aphelios Qwen3 ASR 语音识别 ===".into());
                vm.push(format!("🎤 音频文件：{}", audio_file).into());
                vm.push(format!("🤖 ASR模型：{}", asr_model).into());
                vm.push(format!("🔧 Aligner模型：{}", aligner_model).into());
                vm.push(format!("🔊 VAD模型：{}", vad_model).into());
                vm.push(format!("🌐 语言：{}", language).into());
                vm.push(format!("💾 输出：{}", output_path).into());
            }
            win.invoke_asr_scroll_to_bottom();

            let w2 = w.clone();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let result = rt.block_on(aphelios_asr::qwenasr::qwen3asr_with_vad(
                    &asr_model,
                    &aligner_model,
                    &vad_model,
                    &audio_file,
                    &language,
                ));

                let _ = slint::invoke_from_event_loop(move || {
                    let Some(win) = w2.upgrade() else { return };
                    win.set_asr_is_running(false);
                    match result {
                        Ok(items) => {
                            info!("ASR completed with {} items", items.len());
                            win.set_asr_progress(1.0);
                            win.set_asr_status_message("识别完成!".into());
                            let log_model = win.get_asr_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("✅ 识别完成！共 {} 个词/字", items.len()).into());
                            }
                            if !output_path.is_empty() {
                                if let Ok(content) = std::fs::read_to_string(&output_path) {
                                    win.set_asr_transcription_result(content.into());
                                }
                            }
                        }
                        Err(e) => {
                            error!("ASR failed: {}", e);
                            win.set_asr_progress(0.0);
                            win.set_asr_status_message("识别失败".into());
                            let log_model = win.get_asr_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("❌ 识别失败：{}", e).into());
                            }
                        }
                    }
                    win.invoke_asr_scroll_to_bottom();
                });
            });
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // TTS 页面回调
    // ═══════════════════════════════════════════════════════════════
    let logic_tts = Arc::new(TtsLogic::new(ctx.clone()));

    window.on_tts_text_changed({
        let w = window_weak.clone();
        move |text: slint::SharedString| {
            if let Some(win) = w.upgrade() {
                win.set_tts_input_text_length(text.chars().count() as i32);
            }
        }
    });

    window.on_tts_select_model_path({
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

    window.on_tts_select_output_path({
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

    window.on_tts_select_ref_audio({
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

    window.on_tts_select_txt_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                if !path.is_empty() {
                    let line_count = if let Ok(content) = std::fs::read_to_string(&path) {
                        content.lines().filter(|l| !l.trim().is_empty()).count()
                    } else {
                        0
                    };
                    win.set_tts_txt_file_path(path.into());
                    win.set_tts_txt_line_count(line_count as i32);
                }
            }
        }
    });

    window.on_tts_start_tts({
        let w = window_weak.clone();
        let logic = logic_tts.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let input_text: String = win.get_tts_input_text().to_string();
            let model_path: String = win.get_tts_model_path().to_string();
            let output_path: String = win.get_tts_output_path().to_string();
            let ref_audio_path: String = win.get_tts_ref_audio_path().to_string();
            let ref_text: String = win.get_tts_ref_text().to_string();

            win.set_tts_is_running(true);
            win.set_tts_status_message("TTS 合成中...".into());
            win.set_tts_has_audio(false);
            win.set_tts_progress(0.0);

            let log_model = win.get_tts_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push(format!("🔊 开始 TTS 合成：{}", input_text).into());
                vm.push(format!("🤖 模型路径：{}", model_path).into());
                vm.push(format!("📂 输出路径：{}", output_path).into());
            }
            win.invoke_tts_scroll_to_bottom();

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
                                    win.set_tts_progress(p);
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
                        win.set_tts_is_running(false);
                        match result {
                            Ok(_) => {
                                info!("TTS completed: {}", output_path_inner);
                                win.set_tts_status_message("TTS 完成!".into());
                                win.set_tts_has_audio(true);
                                let log_model = win.get_tts_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push("✅ TTS 合成完成!".into());
                                    win.invoke_tts_scroll_to_bottom();
                                }
                                if let Ok(mut path) = logic_final.ctx().audio_output_path.lock() {
                                    *path = Some(output_path_inner);
                                }
                            }
                            Err(e) => {
                                error!("TTS failed: {}", e);
                                win.set_tts_status_message("TTS 失败".into());
                                let log_model = win.get_tts_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ TTS 失败：{}", e).into());
                                    win.invoke_tts_scroll_to_bottom();
                                }
                            }
                        }
                    });
                },
            );
        }
    });

    window.on_tts_start_batch_tts({
        let w = window_weak.clone();
        let logic = logic_tts.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let txt_file_path: String = win.get_tts_txt_file_path().to_string();
            let model_path: String = win.get_tts_model_path().to_string();
            let ref_audio_path: String = win.get_tts_ref_audio_path().to_string();
            let ref_text: String = win.get_tts_ref_text().to_string();

            win.set_tts_is_running(true);
            win.set_tts_status_message("批量 TTS 合成中...".into());
            win.set_tts_has_audio(false);
            win.set_tts_progress(0.0);

            let log_model = win.get_tts_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push(format!("📄 开始批量 TTS 合成：{}", txt_file_path).into());
                vm.push(format!("🤖 模型路径：{}", model_path).into());
                vm.push("⚙️  批次大小：3，每行一个音频文件".into());
            }
            win.invoke_tts_scroll_to_bottom();

            let w2 = w.clone();
            let logic_inner = logic.clone();
            logic.start_batch_tts(
                txt_file_path,
                model_path,
                ref_audio_path,
                ref_text,
                {
                    let w_progress = w2.clone();
                    move |p| {
                        let _ = slint::invoke_from_event_loop({
                            let w = w_progress.clone();
                            move || {
                                if let Some(win) = w.upgrade() {
                                    win.set_tts_progress(p);
                                }
                            }
                        });
                    }
                },
                move |result| {
                    let w3 = w2.clone();
                    let logic_final = logic_inner.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3.upgrade() else { return };
                        win.set_tts_is_running(false);
                        match result {
                            Ok(output_paths) => {
                                info!("Batch TTS completed: {} files", output_paths.len());
                                win.set_tts_status_message("批量 TTS 完成!".into());
                                win.set_tts_has_audio(true);
                                let log_model = win.get_tts_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(
                                        format!(
                                            "✅ 批量 TTS 合成完成！共 {} 个音频文件",
                                            output_paths.len()
                                        )
                                        .into(),
                                    );
                                    for (i, path) in output_paths.iter().take(5).enumerate() {
                                        vm.push(format!("  [{}] {}", i + 1, path).into());
                                    }
                                    if output_paths.len() > 5 {
                                        vm.push(
                                            format!("  ... 还有 {} 个文件", output_paths.len() - 5)
                                                .into(),
                                        );
                                    }
                                    win.invoke_tts_scroll_to_bottom();
                                }
                                if let Ok(mut path) = logic_final.ctx().audio_output_path.lock() {
                                    if !output_paths.is_empty() {
                                        *path = Some(output_paths[0].clone());
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Batch TTS failed: {}", e);
                                win.set_tts_status_message("批量 TTS 失败".into());
                                let log_model = win.get_tts_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ 批量 TTS 失败：{}", e).into());
                                    win.invoke_tts_scroll_to_bottom();
                                }
                            }
                        }
                    });
                },
            );
        }
    });

    window.on_tts_play_audio({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            if let Some(win) = w.upgrade() {
                if let Ok(path_guard) = ctx.audio_output_path.lock() {
                    if let Some(ref path) = *path_guard {
                        let log_model = win.get_tts_log_messages();
                        if let Some(vm) =
                            log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                        {
                            vm.push(format!("🔊 播放音频：{}", path).into());
                            win.invoke_tts_scroll_to_bottom();
                        }
                        let path_clone = path.clone();
                        std::thread::spawn(move || {
                            play_audio_file(&path_clone);
                        });
                    }
                }
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // 书籍搜索页面回调
    // ═══════════════════════════════════════════════════════════════
    let logic_search = Arc::new(SearchLogic::new(ctx.clone()));

    // 初始化书籍搜索页面数据
    {
        let book_count = logic_search.get_book_count().unwrap_or(0);
        window.set_bs_book_count(book_count as i32);
    }

    window.on_bs_rescan_books({
        let w = window_weak.clone();
        let engine = search_engine_for_rescan.clone();
        let search_client = search_client.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let book_dir = win.get_bs_books_dir().to_string();
            win.set_bs_status_message("正在扫描书籍目录...".into());

            // Update book dir before rescanning
            search_client.set_book_dir(&book_dir);

            let w2 = w.clone();
            let eng = engine.clone();
            std::thread::spawn(move || {
                let result = eng.build_index(None);
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(win) = w2.upgrade() else { return };
                    match result {
                        Ok(count) => {
                            win.set_bs_book_count(count as i32);
                            win.set_bs_status_message(
                                format!("扫描完成，共 {} 本书", count).into(),
                            );
                        }
                        Err(e) => {
                            error!("Rescan failed: {}", e);
                            win.set_bs_status_message(format!("扫描失败: {}", e).into());
                        }
                    }
                });
            });
        }
    });

    window.on_bs_search_books({
        let w = window_weak.clone();
        let logic = logic_search.clone();
        move |query: slint::SharedString| {
            let Some(win) = w.upgrade() else { return };
            win.set_bs_status_message("搜索中...".into());

            let w2 = w.clone();
            logic.search_books_with_mode(
                query.to_string(),
                50,
                SearchMode::Keyword,
                move |result| {
                    let w3 = w2.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3.upgrade() else { return };
                        match result {
                            Ok(search_result) => {
                                let items = search_result.books;
                                let book_items: Vec<BookItem> = items
                                    .iter()
                                    .map(|b| BookItem {
                                        id: b.id.to_string().into(),
                                        title: b.title.clone().into(),
                                        author: b.author.clone().unwrap_or_default().into(),
                                        file_path: b.file_path.clone().into(),
                                        file_type: b.file_type.clone().into(),
                                        file_size: format_file_size(b.file_size).into(),
                                    })
                                    .collect();

                                let model: Rc<VecModel<BookItem>> =
                                    Rc::new(VecModel::from(book_items));
                                win.set_bs_search_results(ModelRc::from(model));
                                win.set_bs_status_message(
                                    format!("找到 {} 个结果", search_result.total).into(),
                                );
                            }
                            Err(e) => {
                                error!("Search failed: {}", e);
                                win.set_bs_status_message(format!("搜索失败: {}", e).into());
                            }
                        }
                    });
                },
            );
        }
    });

    window.on_bs_open_book({
        move |file_path: slint::SharedString| {
            let path: String = file_path.into();
            if !path.is_empty() {
                std::process::Command::new("open").arg(&path).spawn().ok();
            }
        }
    });

    window.on_bs_reveal_in_finder({
        move |file_path: slint::SharedString| {
            let path: String = file_path.into();
            if !path.is_empty() {
                #[cfg(target_os = "macos")]
                {
                    std::process::Command::new("open")
                        .arg("-R")
                        .arg(&path)
                        .spawn()
                        .ok();
                }
                #[cfg(target_os = "windows")]
                {
                    std::process::Command::new("explorer")
                        .arg(format!("/select,{}", path))
                        .spawn()
                        .ok();
                }
                #[cfg(target_os = "linux")]
                {
                    if let Some(parent_dir) = std::path::Path::new(&path).parent() {
                        std::process::Command::new("xdg-open")
                            .arg(parent_dir)
                            .spawn()
                            .ok();
                    }
                }
                #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
                {
                    if let Some(parent_dir) = std::path::Path::new(&path).parent() {
                        std::process::Command::new("xdg-open")
                            .arg(parent_dir)
                            .spawn()
                            .ok();
                    }
                }
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // 文本对齐页面回调
    // ═══════════════════════════════════════════════════════════════
    window.on_ta_select_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ta_model_path(path.into());
            }
        }
    });

    window.on_ta_select_audio_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let audio_path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                if !audio_path.is_empty() {
                    win.set_ta_audio_file_path(audio_path.clone().into());
                    let audio_path_buf = std::path::PathBuf::from(&audio_path);
                    let txt_path = audio_path_buf.with_extension("txt");
                    win.set_ta_target_text_path(txt_path.to_string_lossy().to_string().into());
                    let srt_path = audio_path_buf.with_extension("srt");
                    win.set_ta_output_path(srt_path.to_string_lossy().to_string().into());
                }
            }
        }
    });

    window.on_ta_select_target_text({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("文本文件", &["txt"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ta_target_text_path(path.into());
            }
        }
    });

    window.on_ta_select_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("SRT 文件", &["srt"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_ta_output_path(path.into());
            }
        }
    });

    window.on_ta_start_alignment({
        let w = window_weak.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let model_path: String = win.get_ta_model_path().to_string();
            let audio_file: String = win.get_ta_audio_file_path().to_string();
            let target_text: String = win.get_ta_target_text_path().to_string();
            let output_path: String = win.get_ta_output_path().to_string();
            let min_len: i32 = win.get_ta_min_segment_length().parse().unwrap_or(80);
            let max_len: i32 = win.get_ta_max_segment_length().parse().unwrap_or(120);

            win.set_ta_is_running(true);
            win.set_ta_status_message("正在对齐...".into());

            let log_model = win.get_ta_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push("🎤 开始文本对齐".into());
                vm.push(format!("🤖 模型：{}", model_path).into());
                vm.push(format!("🎵 音频：{}", audio_file).into());
                vm.push(format!("📄 文本：{}", target_text).into());
                vm.push(format!("💾 输出：{}", output_path).into());
                vm.push(format!("⚙️ 分段长度：{} ~ {} 字", min_len, max_len).into());
            }
            win.invoke_ta_scroll_to_bottom();

            let w2 = w.clone();
            let output_path_clone = output_path.clone();
            std::thread::spawn(move || {
                let result = aphelios_asr::text_match::audio_text_match_with_params(
                    &model_path,
                    &audio_file,
                    Some(target_text.as_str()),
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
                    win.set_ta_is_running(false);
                    match result {
                        Ok(srt_path) => {
                            info!("文本对齐完成：{}", srt_path);
                            let log_model = win.get_ta_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(
                                    format!("✅ 对齐完成！SRT 文件已保存到：{}", srt_path).into(),
                                );
                            }
                            win.set_ta_status_message("对齐完成!".into());
                            if let Ok(content) = std::fs::read_to_string(&srt_path) {
                                win.set_ta_srt_preview(content.into());
                                win.invoke_ta_scroll_preview_to_bottom();
                            }
                        }
                        Err(e) => {
                            error!("文本对齐失败：{}", e);
                            let log_model = win.get_ta_log_messages();
                            if let Some(vm) =
                                log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                            {
                                vm.push(format!("❌ 对齐失败：{}", e).into());
                            }
                            win.set_ta_status_message("对齐失败".into());
                        }
                    }
                    win.invoke_ta_scroll_to_bottom();
                });
            });
        }
    });

    window.on_ta_stop_alignment({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                win.set_ta_is_running(false);
                win.set_ta_status_message("已停止".into());
                let log_model = win.get_ta_log_messages();
                if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                    vm.push("⏹️ 已停止".into());
                    win.invoke_ta_scroll_to_bottom();
                }
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // Demucs 页面回调
    // ═══════════════════════════════════════════════════════════════
    let logic_demucs = Arc::new(DemucsLogic::new(ctx.clone()));

    window.on_demucs_select_audio_file({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac", "ogg", "m4a"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_demucs_audio_file_path(path.into());
            }
        }
    });

    window.on_demucs_select_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_demucs_model_path(path.into());
            }
        }
    });

    window.on_demucs_select_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_demucs_output_dir(path.into());
            }
        }
    });

    window.on_demucs_toggle_separation_mode({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let current = win.get_demucs_separation_mode().to_string();
                let new_mode = if current == "vocals_instrumental" {
                    "four_stem"
                } else {
                    "vocals_instrumental"
                };
                win.set_demucs_separation_mode(new_mode.into());
            }
        }
    });

    window.on_demucs_start_separation({
        let w = window_weak.clone();
        let logic = logic_demucs.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            let audio_file: String = win.get_demucs_audio_file_path().to_string();
            let model_path: String = win.get_demucs_model_path().to_string();
            let output_dir: String = win.get_demucs_output_dir().to_string();
            let separation_mode: String = win.get_demucs_separation_mode().to_string();

            let actual_output = if output_dir.is_empty() {
                std::path::Path::new(&audio_file)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string())
            } else {
                output_dir.clone()
            };

            win.set_demucs_is_running(true);
            win.set_demucs_status_message("分离执行中...".into());

            let log_model = win.get_demucs_log_messages();
            if let Some(vm) = log_model.as_any().downcast_ref::<VecModel<SharedString>>() {
                vm.push(format!("🎵 开始 Demucs 分离：{}", audio_file).into());
                vm.push(format!("📂 输出目录：{}", actual_output).into());
                vm.push(format!("🤖 模型路径：{}", model_path).into());
                vm.push(format!("🔧 分离模式：{}", separation_mode).into());
                vm.push("⏳ 正在加载模型并分离...".into());
            }

            let w2 = w.clone();
            let w3 = w.clone();
            logic.start_separation(
                audio_file,
                model_path,
                output_dir,
                separation_mode,
                move |progress| {
                    let w2_clone = w2.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(win) = w2_clone.upgrade() {
                            win.set_demucs_progress(progress);
                        }
                    });
                },
                move |result| {
                    let w3_clone = w3.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        let Some(win) = w3_clone.upgrade() else {
                            return;
                        };
                        match result {
                            Ok(()) => {
                                info!("Demucs separation completed");
                                let log_model = win.get_demucs_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push("✅ 分离完成！文件已保存到输出目录".into());
                                    win.invoke_demucs_scroll_to_bottom();
                                }
                                win.set_demucs_status_message("分离完成!".into());
                            }
                            Err(e) => {
                                error!("Demucs separation failed: {}", e);
                                let log_model = win.get_demucs_log_messages();
                                if let Some(vm) =
                                    log_model.as_any().downcast_ref::<VecModel<SharedString>>()
                                {
                                    vm.push(format!("❌ 分离失败：{}", e).into());
                                    win.invoke_demucs_scroll_to_bottom();
                                }
                                win.set_demucs_status_message("分离失败".into());
                            }
                        }
                        win.set_demucs_is_running(false);
                    });
                },
            );
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // 设置页面回调
    // ═══════════════════════════════════════════════════════════════
    window.on_settings_save({
        let w = window_weak.clone();
        let ctx = ctx.clone();
        move || {
            let Some(win) = w.upgrade() else { return };
            win.set_settings_save_status_message("保存中...".into());

            let mut s = ctx.get_settings();
            s.ocr_model_path = Some(win.get_settings_ocr_model_path().to_string());
            s.ocr_output_dir = Some(win.get_settings_ocr_output_dir().to_string());
            s.asr_model_path = Some(win.get_settings_asr_model_path().to_string());
            s.asr_aligner_model_path = Some(win.get_settings_asr_aligner_model_path().to_string());
            s.asr_vad_model_path = Some(win.get_settings_asr_vad_model_path().to_string());
            s.srt_model_path = Some(win.get_settings_srt_model_path().to_string());
            s.srt_min_segment_length = Some(
                win.get_settings_srt_min_segment_length()
                    .parse()
                    .unwrap_or(80),
            );
            s.srt_max_segment_length = Some(
                win.get_settings_srt_max_segment_length()
                    .parse()
                    .unwrap_or(120),
            );
            s.tts_model_path = Some(win.get_settings_tts_model_path().to_string());
            s.tts_output_path = Some(win.get_settings_tts_output_path().to_string());
            s.tts_ref_audio_path = Some(win.get_settings_tts_ref_audio_path().to_string());
            s.tts_ref_text = Some(win.get_settings_tts_ref_text().to_string());
            s.demucs_model_path = Some(win.get_settings_demucs_model_path().to_string());
            ctx.save_settings(s);

            let w2 = w.clone();
            let w3 = w.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(300));
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(win) = w2.upgrade() {
                        win.set_settings_save_status_message("保存完成".into());
                    }
                });
                std::thread::sleep(std::time::Duration::from_millis(300));
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(win) = w3.upgrade() {
                        win.set_settings_save_status_message("".into());
                    }
                });
            });
        }
    });

    window.on_settings_select_ocr_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_ocr_model_path(path.into());
            }
        }
    });
    window.on_settings_select_ocr_output_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_ocr_output_dir(path.into());
            }
        }
    });
    window.on_settings_select_asr_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_asr_model_path(path.into());
            }
        }
    });
    window.on_settings_select_asr_aligner_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_asr_aligner_model_path(path.into());
            }
        }
    });
    window.on_settings_select_asr_vad_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_asr_vad_model_path(path.into());
            }
        }
    });
    window.on_settings_select_srt_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_srt_model_path(path.into());
            }
        }
    });
    window.on_settings_select_tts_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_tts_model_path(path.into());
            }
        }
    });
    window.on_settings_select_tts_output_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("WAV 音频", &["wav"])
                    .save_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_tts_output_path(path.into());
            }
        }
    });
    window.on_settings_select_tts_ref_audio({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .add_filter("音频文件", &["wav", "mp3", "flac"])
                    .pick_file()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_tts_ref_audio_path(path.into());
            }
        }
    });
    window.on_settings_select_demucs_model_path({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_demucs_model_path(path.into());
            }
        }
    });
    window.on_settings_select_books_dir({
        let w = window_weak.clone();
        move || {
            if let Some(win) = w.upgrade() {
                let path = rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                win.set_settings_books_dir(path.into());
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // 日志订阅（OCR, ASR, TTS, TextAlign, Demucs）
    // ═══════════════════════════════════════════════════════════════
    setup_log_subscription(
        window.as_weak(),
        |win| win.get_ocr_log_messages(),
        |win| win.invoke_ocr_scroll_to_bottom(),
    );
    setup_log_subscription(
        window.as_weak(),
        |win| win.get_asr_log_messages(),
        |win| win.invoke_asr_scroll_to_bottom(),
    );
    setup_log_subscription(
        window.as_weak(),
        |win| win.get_tts_log_messages(),
        |win| win.invoke_tts_scroll_to_bottom(),
    );
    setup_log_subscription(
        window.as_weak(),
        |win| win.get_ta_log_messages(),
        |win| win.invoke_ta_scroll_to_bottom(),
    );
    setup_log_subscription(
        window.as_weak(),
        |win| win.get_demucs_log_messages(),
        |win| win.invoke_demucs_scroll_to_bottom(),
    );

    // 设置初始日志
    {
        // OCR
        let model = window.get_ocr_log_messages();
        if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
            vm.push("=== Aphelios OCR 文字识别 ===".into());
            vm.push("就绪，请选择输入文件开始识别".into());
        }
        window.invoke_ocr_scroll_to_bottom();

        // ASR
        let model = window.get_asr_log_messages();
        if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
            vm.push("=== Aphelios ASR 语音识别 ===".into());
            vm.push("就绪，请选择音频文件开始识别".into());
        }
        window.invoke_asr_scroll_to_bottom();

        // TTS
        let model = window.get_tts_log_messages();
        if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
            vm.push("=== Aphelios TTS 语音合成 ===".into());
            vm.push("就绪，请输入文本开始合成".into());
        }
        window.invoke_tts_scroll_to_bottom();

        // 文本对齐
        let model = window.get_ta_log_messages();
        if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
            vm.push("=== Aphelios 文本对齐 - 生成 SRT 字幕 ===".into());
            vm.push("就绪，请选择音频和目标文本开始对齐".into());
        }
        window.invoke_ta_scroll_to_bottom();

        // Demucs
        let model = window.get_demucs_log_messages();
        if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
            vm.push("=== Aphelios Demucs 人声分离 ===".into());
            vm.push("就绪，请选择音频文件开始分离".into());
        }
        window.invoke_demucs_scroll_to_bottom();
    }

    // ═══════════════════════════════════════════════════════════════
    // 系统托盘
    // ═══════════════════════════════════════════════════════════════
    let open_item = MenuItem::with_id("open-app", "Open App", true, None);
    let quit_item = MenuItem::with_id("quit-app", "Quit", true, None);
    let tray_menu = Menu::with_items(&[&open_item, &quit_item])?;

    let icon_bytes = include_bytes!("../assets/icon.png");
    let icon_img = image::load_from_memory(icon_bytes)?;
    // 缩放到菜单栏图标尺寸（36x36，即 @2x Retina 下的 18x18 点）
    let tray_size = 36u32;
    let small_img = icon_img.resize_exact(tray_size, tray_size, image::imageops::Lanczos3);
    let rgba = small_img.to_rgba8();
    let icon = Icon::from_rgba(rgba.into_raw(), tray_size, tray_size)?;

    let window_for_tray = window.as_weak();
    let open_id = open_item.id().clone();
    let quit_id = quit_item.id().clone();

    MenuEvent::set_event_handler(Some(move |event: tray_icon::menu::MenuEvent| {
        if event.id() == &quit_id {
            let _ = slint::quit_event_loop();
        } else if event.id() == &open_id {
            if let Some(win) = window_for_tray.upgrade() {
                let _ = win.show();
                activate_desktop_app();
            }
        }
    }));

    let _tray_icon = tray_icon::TrayIconBuilder::new()
        .with_icon(icon)
        .with_menu(Box::new(tray_menu))
        .with_tooltip("Aphelios")
        .build()?;

    // ═══════════════════════════════════════════════════════════════
    // 运行
    // ═══════════════════════════════════════════════════════════════
    window.show()?;
    activate_desktop_app();
    slint::run_event_loop_until_quit()?;
    Ok(())
}

// ── 日志订阅辅助函数 ──
fn setup_log_subscription<F, G>(
    window: slint::Weak<AppWindow>,
    get_log_messages: F,
    invoke_scroll: G,
) where
    F: Fn(&AppWindow) -> slint::ModelRc<slint::SharedString> + Clone + Send + 'static,
    G: Fn(&AppWindow) + Clone + Send + 'static,
{
    let mut log_rx = logger::subscribe_logs();
    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.blocking_recv() {
            let w = window.clone();
            let msg_clone = msg.clone();
            let get_log = get_log_messages.clone();
            let invoke_scroll = invoke_scroll.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(win) = w.upgrade() {
                    let model = get_log(&win);
                    if let Some(vm) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
                        vm.push(msg_clone.into());
                        if vm.row_count() > MAX_LOG_LINES {
                            vm.remove(0);
                        }
                        invoke_scroll(&win);
                    }
                }
            });
        }
    });
}

// ── macOS 桌面应用配置 ──

fn initialize_desktop_app() {
    #[cfg(target_os = "macos")]
    {
        configure_macos_app();
    }
}

fn activate_desktop_app() {
    #[cfg(target_os = "macos")]
    {
        activate_macos_app();
    }
}

#[cfg(target_os = "macos")]
fn configure_macos_app() {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };

    let app = NSApplication::sharedApplication(mtm);
    let _ = app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
}

#[cfg(target_os = "macos")]
fn activate_macos_app() {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };

    let app = NSApplication::sharedApplication(mtm);
    let _ = app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    app.activate();
}

// ── 辅助函数 ──

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
    info!("Playing audio file: {}", path);
}
