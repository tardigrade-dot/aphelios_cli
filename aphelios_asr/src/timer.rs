#[macro_export]
macro_rules! measure_time {
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

    ($block:block) => {
        $crate::measure_time!(::std::concat!(::std::file!(), ":", ::std::line!()), $block)
    };

    ($expr:expr) => {
        $crate::measure_time!(::std::concat!(::std::file!(), ":", ::std::line!()), { $expr })
    };
}
