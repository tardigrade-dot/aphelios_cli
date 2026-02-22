//! 性能计时工具
//!
//! 提供函数执行时间测量和性能分析功能

use std::time::Instant;
use tracing::{Level, info};

/// 简单计时器，用于测量代码执行时间
#[derive(Debug)]
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    /// 创建新的计时器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
        }
    }

    /// 获取经过的时间（毫秒）
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// 获取经过的时间（秒）
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// 停止计时器并记录时间
    pub fn stop(self) -> f64 {
        let elapsed = self.elapsed_ms();
        info!("[Timer] {} completed in {:.2}ms", self.name, elapsed);
        elapsed
    }

    /// 停止计时器并记录时间（带额外信息）
    pub fn stop_with_msg(self, msg: &str) -> f64 {
        let elapsed = self.elapsed_ms();
        info!("[Timer] {}: {:.2}ms - {}", self.name, elapsed, msg);
        elapsed
    }
}

/// 作用域计时器，在作用域结束时自动记录时间
pub struct ScopedTimer {
    name: String,
    start: Instant,
    level: Level,
}

impl ScopedTimer {
    /// 创建新的作用域计时器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            level: Level::INFO,
        }
    }

    /// 设置日志级别
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64() * 1000.0;
        match self.level {
            Level::ERROR => tracing::error!("[Timer] {} completed in {:.2}ms", self.name, elapsed),
            Level::WARN => tracing::warn!("[Timer] {} completed in {:.2}ms", self.name, elapsed),
            Level::INFO => info!("[Timer] {} completed in {:.2}ms", self.name, elapsed),
            Level::DEBUG => tracing::debug!("[Timer] {} completed in {:.2}ms", self.name, elapsed),
            Level::TRACE => tracing::trace!("[Timer] {} completed in {:.2}ms", self.name, elapsed),
        }
    }
}

/// 宏：测量代码块执行时间
#[macro_export]
macro_rules! measure_time2 {
    ($name:expr, $block:block) => {{
        let _timer = $crate::utils::ScopedTimer::new($name);
        $block
    }};
}

/// 宏：测量函数执行时间（用于函数开头）
#[macro_export]
macro_rules! profile_function {
    () => {{
        let _timer = $crate::utils::ScopedTimer::new(function_name!());
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test_timer");
        thread::sleep(Duration::from_millis(50));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 50.0 && elapsed < 100.0);
    }

    #[test]
    fn test_scoped_timer() {
        {
            let _timer = ScopedTimer::new("scoped_test");
            thread::sleep(Duration::from_millis(30));
        }
        // Timer should log on drop
    }
}
