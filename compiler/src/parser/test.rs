use crate::parser::grammer::expression::{TypeDef, TypeDefInfoType};

use super::*;

#[test]

fn test_strings() {
    let table = parse(
        "
        const i32 = primative(__int__, 32, true);
        const u32 = primative(__uint__, 32, false);

        const f32 = primative(__float__, 32, true);
        const f64 = primative(__float__, 64, false);

        const bool = primative(__bool__, 32, true);

        const char = primative(__char__, 8, true);

        const foo = fn() {
            let x: &char = \"hi there\";
            // let c: char = '1';
            // let x2 = \"hi there\";

            // if x == x2 {
            //
            //
            // };



        };
        ",
    )
    .expect("Should be vaild");

    eprintln!("HERE I AM: table = {:?}", table);

    table.type_check().expect("all good");
}

#[test]
fn test_type_checking_wrong() {
    let table = parse(
        "
        const i32 = primative(__int__, 32, true);
        const u32 = primative(__uint__, 32, false);

        const f32 = primative(__float__, 32, true);
        const f64 = primative(__float__, 64, false);

        const structFoo = struct {
            a_i32: i32,
            b_u32: u32,
            c_f32: f32
            c_f64: f64

        };

        const structBar = struct {
            foo: structFoo,
            number: u32,
        };

        const foo = fn() {
            let x = 1;

            let bar: i32 = &x;

            let x2: u32 = 32;

            let foo = x + x2;

        };
        ",
    )
    .expect("Should be vaild");

    eprintln!("HERE I AM: table = {:?}", table);

    table.type_check().expect_err("invalid types");
}

#[test]
fn test_type_checking() {
    let table = parse(
        "
        const i32 = primative(__int__, 32, true);
        const u32 = primative(__uint__, 32, false);

        const f32 = primative(__float__, 32, true);
        const f64 = primative(__float__, 64, false);

        const bool = primative(__bool__, 8, true);

        const structFoo = struct {
            a_i32: i32,
            b_u32: u32,
            c_f32: f32
            c_f64: bool

        };

        const structBar = struct {
            foo: structFoo,
            number: u32,
        };

        const foo = fn() {
            let x = 1;
            let x2: u32 = 32;
            let f = 1.;
            let f2: f64 = 1.;

            let foo = x + 1;
            let bar = foo + x;

            let mut y = x;


            let test = foo >= bar;

            let new_val = if test {
                y = x + 10;
                0
            } else if !test {
                y = -x;
                1
            } else {
              2
            };

            let s: structFoo;

            let z: &u32 = &x2;

            let zz: &&u32 = &z;

            let zzz: &&&u32 = &zz;

            let full: u32 = *z;
            let full2: u32 = ***zzz;


        };
        ",
    )
    .expect("is valid");

    table.type_check().expect("type's are valid");

    eprintln!("TABLE = {:?}", table);
}

#[test]
fn test_typing() {
    let val = parse(
        "

        const i32 = primative(__int__, 32, true);
        const u32 = primative(__uint__, 32, false);
        const f32 = primative(__float__, 32, true);
        const f64 = primative(__float__, 64, false);

        const foo = fn() {
            let x = 1;
            let x2: u32 = 32;
            let f = 1.;
            let f2: f64 = 1.;

            let y = &x;

            let z: &i32 = y;

            let zz: i32 = *z ;
            let zzz: i32 = *y ;

        };
        ",
    )
    .expect("is valid");

    eprintln!("val = {:?}", val);
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

    let table = val.expect("is valid");

    assert_eq!(table.len(), 2);

    // first

    let func = match table.get(&"foo".into()) {
        Some(TypeDef {
            size_bits: _,
            type_info: TypeDefInfoType::Function(func),
        }) => func,
        _ => {
            panic!("couldn't find foo as a function");
        }
    };

    assert_eq!(func.args[0].name, "a");
    assert_eq!(func.args[0].datatype, "b");
    assert_eq!(func.args[1].name, "c");
    assert_eq!(func.args[1].datatype, "d");
    assert_eq!(func.args[2].name, "d");
    assert_eq!(func.args[2].datatype, "f");

    assert_eq!((func.ret.as_ref()).expect(""), &"foo");

    let func = match table.get(&"foo2".into()) {
        Some(TypeDef {
            size_bits: _,
            type_info: TypeDefInfoType::Function(func),
        }) => func,
        _ => panic!("couldn't find foo2 as a function"),
    };

    assert_eq!(func.args[0].name, "a");
    assert_eq!(func.args[0].datatype, "b");
    assert_eq!(func.args[1].name, "c");
    assert_eq!(func.args[1].datatype, "d");
    assert_eq!(func.args[2].name, "d");
    assert_eq!(func.args[2].datatype, "f");

    assert_eq!((&func.ret.as_ref()).expect(""), &"bar");
}
