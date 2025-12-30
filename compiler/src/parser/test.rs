use super::*;

#[test]
fn test_function_syntax() {
    assert!(parse(
        "fn foo() {
        let x: y = 1;
        let y: y = \"asdfasfd\";
        let z: y = 334.234;
        let a: y = 'a';
        let b: y = x;
}",
    )
    .is_ok());
}
