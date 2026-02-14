use anyhow::Result;

#[test]
fn usearch_test() -> Result<()> {
    use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

    let options: IndexOptions = IndexOptions {
        dimensions: 3,                  // necessary for most metric kinds
        metric: MetricKind::IP,         // or ::L2sq, ::Cos ...
        quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
        connectivity: 0,                // zero for auto
        expansion_add: 0,               // zero for auto
        expansion_search: 0,
        multi: false,
    };

    let index: Index = new_index(&options).unwrap();

    assert!(index.reserve(10).is_ok());
    assert!(index.capacity() >= 10);
    assert!(index.connectivity() != 0);
    assert_eq!(index.dimensions(), 3);
    assert_eq!(index.size(), 0);

    let first: [f32; 3] = [0.2, 0.1, 0.2];
    let second: [f32; 3] = [0.2, 0.1, 0.2];

    assert!(index.add(42, &first).is_ok());
    assert!(index.add(43, &second).is_ok());
    assert_eq!(index.size(), 2);

    // Read back the tags
    let results = index.search(&first, 10).unwrap();
    assert_eq!(results.keys.len(), 2);

    assert!(
        index
            .save("/Users/larry/coderesp/aphelios_cli/output/index.usearch")
            .is_ok()
    );
    assert!(
        index
            .load("/Users/larry/coderesp/aphelios_cli/output/index.usearch")
            .is_ok()
    );
    assert!(
        index
            .view("/Users/larry/coderesp/aphelios_cli/output/index.usearch")
            .is_ok()
    );
    Ok(())
}
