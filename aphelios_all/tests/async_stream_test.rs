use async_stream::stream;

use futures_util::stream::StreamExt;
use futures_util::{Stream, pin_mut};
use tokio::sync::futures;
use tracing::{Level, info};

#[tokio::test]
#[tracing_test::traced_test]
async fn main_test() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .try_init();

    let s = stream! {
        for i in 0..3 {
            yield i;
        }
    };
    pin_mut!(s);
    s_process(s).await;
}

async fn s_process<S>(s: S)
where
    S: Stream<Item = i32>,
{
    pin_mut!(s);

    while let Some(value) = s.next().await {
        println!("got {}", value);
    }
}
