use alphawinnow::{analyze_expression, canonical, parse_expression, semantic_fingerprint};

#[test]
fn semantic_identity_fixture_is_golden() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/semantic_identity.json"))
            .expect("valid golden fixture");
    for pair in fixture["equivalent"].as_array().unwrap() {
        let left = parse_expression(pair[0].as_str().unwrap()).unwrap();
        let right = parse_expression(pair[1].as_str().unwrap()).unwrap();
        assert_eq!(
            semantic_fingerprint(&left),
            semantic_fingerprint(&right),
            "expected equivalent: {} and {}",
            canonical(&left),
            canonical(&right)
        );
    }
    for pair in fixture["distinct"].as_array().unwrap() {
        let left = parse_expression(pair[0].as_str().unwrap()).unwrap();
        let right = parse_expression(pair[1].as_str().unwrap()).unwrap();
        assert_ne!(
            semantic_fingerprint(&left),
            semantic_fingerprint(&right),
            "expected distinct: {} and {}",
            canonical(&left),
            canonical(&right)
        );
    }
    for rejected in fixture["rejected"].as_array().unwrap() {
        let expression = parse_expression(rejected["expression"].as_str().unwrap()).unwrap();
        let analysis = analyze_expression(&expression);
        let reason = serde_json::to_value(analysis.rejection_reason.unwrap()).unwrap();
        assert_eq!(reason, rejected["reason"]);
    }
}
