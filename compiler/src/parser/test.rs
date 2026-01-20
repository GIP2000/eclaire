use crate::parser::grammer::expression::{
    ConstantExpression, Expression, TypeDef, TypeDefInfoType,
};

use super::*;

#[test]
fn test_typing() {
    let val = parse(
        "

        const i32 = primative(int, 32, true);
        const u32 = primative(uint, 32, false);
        const f32 = primative(float, 32, true);
        const f64 = primative(float, 64, false);

        const foo = fn() {

            let x = 1;
            let x2: u32 = 32;
            let f = 1.;
            let f2: f64 = 1.;

        }
        ",
    )
    .expect("is valid");

    eprint!("val = {:?}", val);

    assert!(false);
}

#[test]
fn test_wrong_function_syntax() {
    let val = parse(
        "
        const foo = fn(a:b, c:d, d:2f) -> foo {};
        ",
    )
    .expect_err("This pattern should be invalid");

    assert!(matches!(
        val,
        ParserError::LexerError(LexerIteratorError::DoesNotMatch(_))
    ))
}

#[test]
fn test_function_syntax() {
    let val = parse(
        "
        const foo = fn (a:b, c:d, d:f) -> foo {
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
        };
        const foo2 = fn (a:b, c:d, d:f) -> bar {};
        ",
    );

    eprintln!("val: {val:?}");

    let (val, _) = val.expect("is valid");

    assert_eq!(val.0.len(), 2);

    // first

    assert_eq!(val.0[0].ident, "foo");

    let func = match &val.0[0].expr {
        Some(Expression::Constant(ConstantExpression::TypeLit(TypeDef {
            size_bits: _,
            type_info: TypeDefInfoType::Function(func),
        }))) => func,
        _ => {
            assert!(false, "const foo is not a function");
            unreachable!("")
        }
    };

    assert_eq!(func.args[0].name, "a");
    assert_eq!(func.args[0].datatype, "b");
    assert_eq!(func.args[1].name, "c");
    assert_eq!(func.args[1].datatype, "d");
    assert_eq!(func.args[2].name, "d");
    assert_eq!(func.args[2].datatype, "f");

    assert_eq!((func.ret.as_ref()).expect(""), &"foo");

    // second

    assert_eq!(val.0[1].ident, "foo2");

    let func = match &val.0[1].expr {
        Some(Expression::Constant(ConstantExpression::TypeLit(TypeDef {
            size_bits: _,
            type_info: TypeDefInfoType::Function(func),
        }))) => func,
        _ => {
            assert!(false, "const foo is not a function");
            unreachable!("")
        }
    };

    assert_eq!(func.args[0].name, "a");
    assert_eq!(func.args[0].datatype, "b");
    assert_eq!(func.args[1].name, "c");
    assert_eq!(func.args[1].datatype, "d");
    assert_eq!(func.args[2].name, "d");
    assert_eq!(func.args[2].datatype, "f");

    assert_eq!((&func.ret.as_ref()).expect(""), &"bar");
}
