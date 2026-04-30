#[macro_export]
macro_rules! measure_time {
    // ----------------------------------------------------------------
    // 规则 1：显式 desc + 花括号块
    //   measure_time!("desc", { stmt; stmt; expr })
    // ----------------------------------------------------------------
    ($desc:expr, $block:block) => {{
        #[cfg(feature = "profiling")]
        {
            let __start = ::std::time::Instant::now();
            ::tracing::info!("[profiling] >>>>> start {}", $desc);
            let __result = $block;
            ::tracing::info!(
                "[profiling] <<<<< {} cost [{:.3}s]",
                $desc,
                __start.elapsed().as_secs_f64()
            );
            __result
        }
        #[cfg(not(feature = "profiling"))]
        {
            $block
        }
    }};

    // ----------------------------------------------------------------
    // 规则 2：显式 desc + 单表达式（含 `?`）
    //   measure_time!("desc", expr?)
    // ----------------------------------------------------------------
    ($desc:expr, $expr:expr) => {{
        #[cfg(feature = "profiling")]
        {
            let __start = ::std::time::Instant::now();
            ::tracing::info!("[profiling] >>>>> start {}", $desc);
            let __result = $expr;
            ::tracing::info!(
                "[profiling] <<<<< {} cost [{:.3}s]",
                $desc,
                __start.elapsed().as_secs_f64()
            );
            __result
        }
        #[cfg(not(feature = "profiling"))]
        {
            $expr
        }
    }};

    // ----------------------------------------------------------------
    // 规则 3：无 desc + 花括号块（自动 file:line）
    //   measure_time!({ stmt; stmt; expr })
    // ----------------------------------------------------------------
    ($block:block) => {
        $crate::measure_time!(::std::concat!(::std::file!(), ":", ::std::line!()), $block)
    };

    // ----------------------------------------------------------------
    // 规则 4：无 desc + 单表达式（自动 file:line）
    //   measure_time!(expr?)
    // ----------------------------------------------------------------
    ($expr:expr) => {
        $crate::measure_time!(::std::concat!(::std::file!(), ":", ::std::line!()), {
            $expr
        })
    };
}

#[macro_export]
macro_rules! profile_only {
    ($desc:expr, $block:block) => {{
        #[cfg(feature = "profiling")]
        {
            let __start = ::std::time::Instant::now();
            {
                $block
            };
            ::tracing::info!(
                "[profiling] ✓ {} -> {:.3}s",
                $desc,
                __start.elapsed().as_secs_f64()
            );
        }
    }};

    ($block:block) => {
        $crate::profile_only!(::std::concat!(::std::file!(), ":", ::std::line!()), $block)
    };
}

pub struct ScopedTimer {
    name: &'static str,
    start: ::std::time::Instant,
}

impl ScopedTimer {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: ::std::time::Instant::now(),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        ::tracing::info!(
            "[Timer] {} -> {:.3}s",
            self.name,
            self.start.elapsed().as_secs_f64()
        );
    }
}
