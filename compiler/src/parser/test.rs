use super::*;

#[test]
fn test_wrong_function_syntax() {
    let val = parse(
        "
        fn foo(a:b, c:d, d:2f) -> foo {}
        ",
    )
    .expect_err("This pattern should be invalid");

    assert!(matches!(
        val,
        ParserError::LexerError(LexerIteratorError::DoesNotMatch)
    ))
}

#[test]
fn test_function_syntax() {
    let val = parse(
        "
        fn foo(a:b, c:d, d:f) -> foo {
            let x = 2;
            let y = z;
            let z = 12.;
            let z = 12.2;
            x;
            y;
            \"hi\";
            123;
            '1';
            12.34;
        }
        fn foo2(a:b, c:d, d:f) -> bar {}
        ",
    )
    .expect("is valid");

    assert_eq!(val.functions.len(), 2);

    // first

    assert_eq!(val.functions[0].name, "foo");

    assert_eq!(val.functions[0].args[0].name, "a");
    assert_eq!(val.functions[0].args[0].datatype, "b");
    assert_eq!(val.functions[0].args[1].name, "c");
    assert_eq!(val.functions[0].args[1].datatype, "d");
    assert_eq!(val.functions[0].args[2].name, "d");
    assert_eq!(val.functions[0].args[2].datatype, "f");

    assert_eq!((&val.functions[0].ret.as_ref()).expect(""), &"foo");

    // second

    assert_eq!(val.functions[1].name, "foo2");

    assert_eq!(val.functions[1].args[0].name, "a");
    assert_eq!(val.functions[1].args[0].datatype, "b");
    assert_eq!(val.functions[1].args[1].name, "c");
    assert_eq!(val.functions[1].args[1].datatype, "d");
    assert_eq!(val.functions[1].args[2].name, "d");
    assert_eq!(val.functions[1].args[2].datatype, "f");

    assert_eq!((&val.functions[1].ret.as_ref()).expect(""), &"bar");
}
